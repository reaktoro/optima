/*
 * AlgorithmIPFilter.cpp
 *
 *  Created on: 4 Apr 2013
 *      Author: allan
 */

#include "AlgorithmIPFilter.hpp"

// C++ includes
#include <cmath>
#include <limits>

// Optima includes
#include "MathUtils.hpp"

namespace Optima {
namespace IPFilter {

/// The double representation of infinity
const double infinity = std::numeric_limits<double>::infinity();

/// Transform a number in a positive number using the formula: a if a > 0, else infinity
inline double MakePositive(double a)
{
    return (a > 0.0) ? a : infinity;
}

/// Calculates the step length t such that p + t*dp touches the boundary
double CalculateLargestBoundaryStep(const VectorXd& p, const VectorXd& dp)
{
    const auto& zero = VectorXd::Zero(p.rows());

    const double step = -p.cwiseQuotient(dp).cwiseMin(zero).maxCoeff();

    return MakePositive(step);
}

/// Calculate the step length t that solves the family of quadratic functions f_i(t)
double CalculateLargestQuadraticStep(const VectorXd& a, const VectorXd& b, const VectorXd& c, const VectorXd& d, unsigned n, double gamma)
{
    const VectorXd a1 = b.array() * d.array() * n - b.dot(d) * gamma;
    const VectorXd a2 = (a.array() * d.array() + b.array() * c.array()) * n - (a.dot(d) + b.dot(c)) * gamma;
    const VectorXd a3 = a.array() * c.array() * n - a.dot(c) * gamma;

    auto solve = [](double a, double b, double c) -> double
    {
        const double aux = std::sqrt(b*b - 4*a*c);

        const double r1 = (-b + aux)/(2*a);
        const double r2 = (-b - aux)/(2*a);

        return std::min(MakePositive(r1), MakePositive(r2));
    };

    double step = solve(a1[0], a2[0], a3[0]);
    for(unsigned i = 1; i < n; ++i)
        step = std::min(step, solve(a1[i], a2[i], a3[i]));

    return step;
}

bool AnyFloatingPointException(const ObjectiveState& f)
{
    if(isfinite(f.func) and isfinite(f.grad) and isfinite(f.hessian)) return false;
    else return true;
}

bool AnyFloatingPointException(const ConstraintState& h)
{
    if(isfinite(h.func) and isfinite(h.grad)) return false;
    else return true;
}

AlgorithmIPFilter::State::State(const OptimumProblem& problem, const Params& params, const Options& options)
: dimx(problem.dim_objective), dimy(problem.dim_constraint),
  problem(problem), params(params), options(options)
{
    const unsigned dim = dimx + dimy;

    lhs.resize(dim, dim);
    rhs.resize(dim);
}

void AlgorithmIPFilter::State::Initialise(const VectorXd& x_, const VectorXd& y_, const VectorXd& z_)
{
    x = x_old = x_;
    y = y_old = y_;
    z = z_old = z_;

    f = f_old = problem.objective(x_old);
    h = h_old = problem.constraint(x_old);





    // Initialise the value of the parameter gamma
    gamma = std::min(params.gamma_min, x.cwiseProduct(z).minCoeff()/(2.0*mu));

    // Initialise the value of the parameter c
    const double sig = params.sigma_slow;
    c = 3 * dimx/(1 - sig) * std::pow(std::max(1.0, (1 - sig)/gamma), 2);

    // Initialise the value of the neighborhood parameter M
    const double Mcirc = params.neighM_max;
    const double alphM = params.alphaM;
    M = std::max(Mcirc, alphM*(thh + thl)/mu);
}

double AlgorithmIPFilter::State::CalculateLargestDelta() const
{
    // Define some auxiliary variables
    const auto& n    = problem.dim_objective;
    const auto& ones = VectorXd::Ones(n);

    // The vectors x(delta) and z(delta) assuming that both alpha_n and alpha_t are 1
    const VectorXd xcirc = x_old + snx + stx;
    const VectorXd zcirc = z_old + snz + stz;

    // Check if the Trivial Case is satisfied, where alpha_n = alpha_n = 1 yields positivity condition
    if(xcirc.minCoeff() > 0 and zcirc.minCoeff() > 0)
        if((xcirc.cwiseProduct(zcirc) - gamma/n * xcirc.dot(zcirc) * ones).minCoeff() > 0.0)
            return infinity;

    // Calculate the auxiliary vectors a and b for Case I
    const VectorXd aI = x_old + snx;
    const VectorXd bI = stx/norm_st;
    const VectorXd cI = z_old + snz;
    const VectorXd dI = stz/norm_st;

    // Calculate the auxiliary vectors a and b for Case II
    const VectorXd&aII = x_old;
    const VectorXd bII = snx/norm_sn + stx/std::max(norm_sn, norm_st);
    const VectorXd&cII = z_old;
    const VectorXd dII = snz/norm_sn + stz/std::max(norm_sn, norm_st);

    // Calculate the auxiliary delta values for Case I
    const double delta_xI  = CalculateLargestBoundaryStep( aI,  bI);
    const double delta_zI  = CalculateLargestBoundaryStep( cI,  dI);
    const double delta_xII = CalculateLargestBoundaryStep(aII, bII);
    const double delta_zII = CalculateLargestBoundaryStep(cII, dII);

    // Calculate the auxiliary delta values for Case II
    const double delta_xzI  = CalculateLargestQuadraticStep( aI,  bI,  cI,  dI, n, gamma);
    const double delta_xzII = CalculateLargestQuadraticStep(aII, bII, cII, dII, n, gamma);

    // Calculate the minimum among all auxiliary delta values for Case I and II
    const double delta = std::min({delta_xI, delta_zI, delta_xII, delta_zII, delta_xzI, delta_xzII});

    // Calculate the adjustment factor tau
    const double tau = 1.0 - std::min(0.01, 100.0*mu_old*mu_old);

    return tau*delta;
}

void AlgorithmIPFilter::State::SearchDeltaNeighborhood() throw(SearchDeltaNeighborhoodError)
{
    // Calculate the largest delta that solves the positivity conditions
    const double delta_largest = CalculateLargestDelta();

    // Calculate the start trial delta so that it is not greater than the current maximum allowed
    double delta_trial = std::min(delta_largest, delta_max);

    while(true)
    {
        // Update the members that are dependent on delta
        UpdateDelta(delta_trial);

        // Decrease the current value of the trial delta
        delta_trial *= params.delta_decrease_factor;

        // Check if delta is now less than the allowed minimum
        if(delta_trial < params.delta_min)
            throw SearchDeltaNeighborhoodError();

        // Check if the current delta results results in any IEEE floating point exception
        if(AnyFloatingPointException(f) or AnyFloatingPointException(h))
            continue;

        // Check if the current delta results in a point (x,y,z) that pass the centrality neighborhood condition
        if(thh + thl <= M*mu)
            break;
    }
}

void AlgorithmIPFilter::State::SearchDelta() throw(SearchDeltaError)
{
    while(true)
    {
        // Check if the current trust-region radius is less than the allowed minimum
        if(delta < params.delta_min)
            throw SearchDeltaError();

        if(m_old - m < params.kappa*theta_old*theta_old)
        {
            const double beta_theta = theta_old * (1 - params.alpha_theta);
            const double beta_psi   = psi_old - params.alpha_psi*theta_old;

            if((theta < beta_theta or psi < beta_psi) and filter.IsAcceptable({theta, psi}))
            {
                // Increase the current maximum trust-region radius
                delta_max *= params.delta_increase_factor;

                // Add a new entry to the filter
                filter.Add({beta_theta, beta_psi});

                // Reset the Lagrange multipliers z
                const double aux1 = mu*params.kappa_zreset;
                const double aux2 = mu/params.kappa_zreset;
                z = z.array().min(aux1/x.array()).max(aux2/x.array());

                // Update the neighborhood parameter M
                if(thh + thl > mu * params.epsilonM * M)
                    M = std::max(params.neighM_max, params.alphaM*(thh + thl)/mu);

                // Trial point has been found: leave loop
                break;
            }
        }
        else
        {
            // Calculate the ratio of decrease in actual and predicted optimality
            const double rho = (psi_old - psi)/(m_old - m);

            if(rho > params.eta_small and filter.IsAcceptable({theta, psi}))
            {
                // Increase the current maximum trust-region radius
                if(rho > params.eta_large) delta_max *= params.delta_increase_factor;

                // Reset the Lagrange multipliers z
                const double aux1 = mu*params.kappa_zreset;
                const double aux2 = mu/params.kappa_zreset;
                z = z.array().min(aux1/x.array()).max(aux2/x.array());

                // Update the neighborhood parameter M
                if(thh + thl > mu * params.epsilonM * M)
                    M = std::max(params.neighM_max, params.alphaM*(thh + thl)/mu);

                // Trial point has been found: leave loop
                break;
            }
        }

        // Decrease the trust-region radius
        UpdateDelta(params.delta_decrease_factor * delta);
    }
}

void AlgorithmIPFilter::State::SearchDeltaRestoration() throw(SearchDeltaRestorationError)
{
    // Calculate the optimality measure of the restoration algorithm at w
    const double theta2_old = (thh_old*thh_old + thc_old*thc_old)/2.0;

    // Calculate the d/dx and d/dz derivatives of 1/2*thh^2 and 1/2*thc^2
    const auto& ddx_thh2 = h_old.grad.transpose() * h_old.func;
    const auto& ddx_thc2 = (z_old.array() * (x_old.array() * z_old.array() - mu_old)).matrix();
    const auto& ddz_thc2 = (x_old.array() * (x_old.array() * z_old.array() - mu_old)).matrix();

    // Calculate the dot product of grad(theta2) with the normal step sn
    const double grad_theta2_dot_sn = (ddx_thh2 + ddx_thc2).dot(snx) + ddz_thc2.dot(snz);

    while(true)
    {
        // Calculate the optimality measure of the restoration algorithm at w(delta)
        const double theta2 = (thh*thh + thc*thc)/2.0;

        // Calculate the ratio of decrease in actual and predicted of theta2
        const double rho = (theta2_old - theta2)/(-grad_theta2_dot_sn);

        // Increase the current maximum trust-region radius
        if(rho > params.xi2) delta_max = delta * params.delta_increase_factor;

        // Check if rho satisfies the Cauchy condition of the restoration algorithm
        if(rho > params.xi1)
            break;

        // Decrease the current trust-region radius and update all delta dependent quantities
        UpdateDelta(delta * params.delta_decrease_factor);

        // Check if delta is now less than the allowed minimum
        if(delta < params.delta_min)
            throw SearchDeltaRestorationError();
    }
}

void AlgorithmIPFilter::State::UpdateDelta(double value)
{
    // Update the delta member
    delta = value;

    // Update the normal and tangencial step lengths
    alpha_n = std::min(1.0, delta/norm_sn);
    alpha_t = std::min(alpha_n, delta/norm_st);

    // Update the iterates x(delta), y(delta), and z(delta)
    x = x_old + alpha_n * snx + alpha_t * stx;
    y = y_old + alpha_n * sny + alpha_t * sty;
    z = z_old + alpha_n * snz + alpha_t * stz;

    // Update the objective and constraint state at x(delta)
    f = problem.objective(x);
    h = problem.constraint(x);

    // Update the barrier parameter at (x(delta),y(delta),z(delta))
    mu = x.dot(z)/dimx;

    // Update the auxiliary optimality theta measures at (x(delta),y(delta),z(delta))
    thc = (x.array() * z.array() - mu).matrix().norm();
    thh = (h.func).norm();
    thl = (f.grad + h.grad.transpose() * y - z).norm();

    // Update the feasibility/centrality theta measure at (x(delta),y(delta),z(delta))
    theta = thh + thc;

    // Update the optimality psi measure at (x(delta),y(delta),z(delta))
    psi = CalculatePsi();

    // Update the linear model m(w(Delta))
    m = CalculateLinearModel();
}

void AlgorithmIPFilter::State::AcceptTrialPoint()
{
    x_old     = x;
    y_old     = y;
    z_old     = z;
    f_old     = f;
    h_old     = h;
    mu_old    = mu;
    thh_old   = thh;
    thc_old   = thc;
    thl_old   = thl;
    theta_old = theta;
    psi_old   = psi;
    m_old     = m;

    ++iter;
}

void AlgorithmIPFilter::State::ComputeSteps()
{
    // Define some auxiliary variables
    const unsigned n = dimx;
    const unsigned m = dimy;

    // Calculate the matrix H, which is the block(1,1) of the linear system
    H = f_old.hessian;

    for(unsigned i = 0; i < h_old.hessian.size(); ++i)
        H += y_old[i] * h_old.hessian[i];

    H += z_old.cwiseQuotient(x_old).asDiagonal();

    // Assemble the coefficient matrix of the linear system
    lhs.block(0, 0, n, n) = H;
    lhs.block(0, n, n, m) = h_old.grad.transpose();
    lhs.block(n, 0, m, n) = h_old.grad;
    lhs.block(n, n, m, m) = MatrixXd::Identity(m, m);

    // Calculate the LU decomposition of the coefficient matrix
    lu.compute(lhs);

    // Assemble the normal rhs vector of the linear system
    rhs.segment(0, n) = - z_old + (mu_old/x_old.array()).matrix();
    rhs.segment(n, m) = - h_old.func;

    // Calculate the normal step
    u = lu.solve(rhs);

    // Extract the x and y components of the normal step
    snx = u.segment(0, n);
    sny = u.segment(n, m);

    // Assemble the tangential rhs vector of the linear system
    rhs.segment(0, n) = - f_old.grad - h_old.grad.transpose()*y_old + z_old - ((1 - sigma)*mu_old/x.array()).matrix();
    rhs.segment(n, m) = - VectorXd::Zero(m);

    // Calculate the tangential step
    u = lu.solve(rhs);

    // Extract the x and y components of the tangential step
    stx = u.segment(0, n);
    sty = u.segment(n, m);

    // Calculate the z components of the normal and tangential steps
    snz = -(z_old.array()*snx.array() + x_old.array()*z_old.array() - mu_old)/x_old.array();
    stz = -(z_old.array()*stx.array() + mu_old*(1 - sigma))/x_old.array();

    // Calculate the norms of the normal and tangential steps
    norm_sn = std::sqrt(snx.squaredNorm() + sny.squaredNorm() + snz.squaredNorm());
    norm_st = std::sqrt(stx.squaredNorm() + sty.squaredNorm() + stz.squaredNorm());
}

void AlgorithmIPFilter::State::SolveRestoration() throw(MaxIterationError)
{
    while(true)
    {
        SearchDeltaRestoration();

        if(PassRestorationCondition() and PassFilterCondition())
            break;

        AcceptTrialPoint();

        ComputeSteps();

        SearchDeltaNeighborhood();

        if(iter > options.max_iter)
            throw MaxIterationError();
    }
}

void AlgorithmIPFilter::State::Solve() throw(MaxIterationError)
{
    while(true)
    {
        ComputeSteps();

        SearchDeltaNeighborhood();

        if(PassRestorationCondition()) SearchDelta();
        else SolveRestoration();

        AcceptTrialPoint();

        if(PassStoppingCriteria()) break;
    }
}

double AlgorithmIPFilter::State::CalculatePsi() const
{
    switch(options.psi)
    {
    case Objective:
        return f.func + c * mu;
    case Lagrange:
        return f.func + h.func.dot(y) + c * mu;
    default:
        return f.func + c * mu;
    }
}

double AlgorithmIPFilter::State::CalculateLinearModel() const
{
    switch(options.psi)
    {
    case Objective:
        return psi_old + (f_old.grad + c/dimx*z_old).dot(x - x_old) + c/dimx*x_old.dot(z - z_old);
    case Lagrange:
        return psi_old + (f_old.grad + h_old.grad.transpose()*y_old + c/dimx*z_old).dot(x - x_old) + h.func.dot(y - y_old) + c/dimx*x_old.dot(z - z_old);
    default:
        return psi_old + (f_old.grad + c/dimx*z_old).dot(x - x_old) + c/dimx*x_old.dot(z - z_old);
    }
}

bool AlgorithmIPFilter::State::PassStoppingCriteria() const
{
    const double sc = 0.01 * std::max(100.0, z.lpNorm<1>()/dimx);
    const double sl = 0.01 * std::max(100.0, (y.lpNorm<1>() + z.lpNorm<1>())/(dimx + dimy));

    const double opt1 = (f.grad + h.grad.transpose()*y - z).lpNorm<Infinity>();
    const double opt2 = h.func.lpNorm<Infinity>();
    const double opt3 = (x.array() * z.array() - mu).matrix().lpNorm<Infinity>();

    const double error = std::max({opt1/sl, opt2, opt3/sc});

    return error < options.tolerance;
}

bool AlgorithmIPFilter::State::PassRestorationCondition() const
{
    return theta <= delta * std::min(params.gamma1, params.gamma2*std::pow(delta, params.beta));
}

bool AlgorithmIPFilter::State::PassFilterCondition() const
{
    return filter.IsAcceptable({theta, psi});
}

} /* namespace IPFilter */
} /* namespace Optima */
