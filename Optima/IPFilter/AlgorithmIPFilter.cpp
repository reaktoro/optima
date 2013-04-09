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
namespace {

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
double CalculateLargestQuadraticStep(
    const VectorXd& a, const VectorXd& b,
    const VectorXd& c, const VectorXd& d, unsigned n, double gamma)
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

} /* namespace */

AlgorithmIPFilter::State::State()
{}

AlgorithmIPFilter::AlgorithmIPFilter()
{}

void AlgorithmIPFilter::SetParams(const Params& params)
{
    this->params = params;
}

void AlgorithmIPFilter::SetOptions(const Options& options)
{
    this->options = options;
}

void AlgorithmIPFilter::SetProblem(const OptimumProblem& problem)
{
    this->problem = problem;

    dimx = problem.dim_objective;
    dimy = problem.dim_constraint;

    const unsigned dim = dimx + dimy;

    lhs.resize(dim, dim);
    rhs.resize(dim);
}

void AlgorithmIPFilter::Initialise(const State& state)
{
    // Initialise the current and next states
    curr = next = state;

    // Initialise the value of the parameter gamma
    gamma = std::min(params.gamma_min, curr.x.cwiseProduct(curr.z).minCoeff()/(2.0*curr.mu));

    // Initialise the value of the parameter c
    c = 3*dimx/(1 - params.sigma_slow)*std::pow(std::max(1.0, (1 - params.sigma_slow)/gamma), 2);

    // Initialise the value of the neighborhood parameter M
    M = std::max(params.neighM_max, params.alphaM*(curr.thh + curr.thl)/curr.mu);

    // Deactivate the restoration flag
    restoration = false;
}

void AlgorithmIPFilter::Solve()
{
    while(true)
    {
        // Calculate the normal and tangential steps for the trust-region algorithm
        UpdateNormalTangentialSteps();

        // Search for a trust-region radius that satisfies the centrality neighborhood conditions
        SearchDeltaNeighborhood();

        // If the current state pass the restoration condition, search for a suitable trust-region radius
        if(PassRestorationCondition()) SearchDeltaTrustRegion();

        // Otherwise, start the restoration algorithm that finds a suitable (x,y,z)
        else SolveRestoration();

        // Accept the current trial point
        AcceptTrialPoint();

        // Check if the current state pass the stopping criteria of optimality
        if(PassStoppingCriteria()) break;
    }
}

bool AlgorithmIPFilter::AnyFloatingPointException(const State& state) const
{
    if(isfinite(state.f.func) and isfinite(state.f.grad) and isfinite(state.f.hessian))
        if(isfinite(state.h.func) and isfinite(state.h.grad)) return false;
    return true;
}

bool AlgorithmIPFilter::PassFilterCondition() const
{
    return filter.IsAcceptable({next.theta, next.psi});
}

bool AlgorithmIPFilter::PassRestorationCondition() const
{
    return curr.theta <= delta * std::min(params.gamma1, params.gamma2*std::pow(delta, params.beta));
}

bool AlgorithmIPFilter::PassStoppingCriteria() const
{
    const double sc = 0.01 * std::max(100.0, next.z.lpNorm<1>()/dimx);
    const double sl = 0.01 * std::max(100.0, (next.y.lpNorm<1>() + next.z.lpNorm<1>())/(dimx + dimy));

    const double opt1 = (next.f.grad + next.h.grad.transpose()*next.y - next.z).lpNorm<Infinity>();
    const double opt2 = next.h.func.lpNorm<Infinity>();
    const double opt3 = (next.x.array() * next.z.array() - next.mu).matrix().lpNorm<Infinity>();

    const double error = std::max({opt1/sl, opt2, opt3/sc});

    return error < options.tolerance;
}

double AlgorithmIPFilter::CalculateDeltaPositivity() const
{
    // Define some auxiliary variables
    const auto& ones = VectorXd::Ones(dimx);

    // The vectors x(delta) and z(delta) assuming that both alpha_n and alpha_t are 1
    const VectorXd xcirc = curr.x + snx + stx;
    const VectorXd zcirc = curr.z + snz + stz;

    // Check if the Trivial Case is satisfied, where alpha_n = alpha_n = 1 yields positivity condition
    if(xcirc.minCoeff() > 0 and zcirc.minCoeff() > 0)
        if((xcirc.cwiseProduct(zcirc) - gamma/dimx * xcirc.dot(zcirc) * ones).minCoeff() > 0.0)
            return infinity;

    // Calculate the auxiliary vectors a and b for Case I
    const VectorXd aI = curr.x + snx;
    const VectorXd bI = stx/norm_st;
    const VectorXd cI = curr.z + snz;
    const VectorXd dI = stz/norm_st;

    // Calculate the auxiliary vectors a and b for Case II
    const VectorXd&aII = curr.x;
    const VectorXd bII = snx/norm_sn + stx/std::max(norm_sn, norm_st);
    const VectorXd&cII = curr.z;
    const VectorXd dII = snz/norm_sn + stz/std::max(norm_sn, norm_st);

    // Calculate the auxiliary delta values for Case I
    const double delta_xI  = CalculateLargestBoundaryStep( aI,  bI);
    const double delta_zI  = CalculateLargestBoundaryStep( cI,  dI);
    const double delta_xII = CalculateLargestBoundaryStep(aII, bII);
    const double delta_zII = CalculateLargestBoundaryStep(cII, dII);

    // Calculate the auxiliary delta values for Case II
    const double delta_xzI  = CalculateLargestQuadraticStep( aI,  bI,  cI,  dI, dimx, gamma);
    const double delta_xzII = CalculateLargestQuadraticStep(aII, bII, cII, dII, dimx, gamma);

    // Calculate the minimum among all auxiliary delta values for Case I and II
    const double aux_delta = std::min({delta_xI, delta_zI, delta_xII, delta_zII, delta_xzI, delta_xzII});

    // Calculate the adjustment factor tau
    const double tau = 1.0 - std::min(0.01, 100.0 * curr.mu * curr.mu);

    return tau*aux_delta;
}

double AlgorithmIPFilter::CalculateNextLinearModel() const
{
    if(options.psi_scheme == 0)
        return curr.psi + (curr.f.grad + c/dimx*curr.z).dot(next.x - curr.x) + c/dimx*curr.x.dot(next.z - curr.z);
    else
        return curr.psi + (curr.f.grad + curr.h.grad.transpose()*curr.y + c/dimx*curr.z).dot(next.x - curr.x) +
            curr.h.func.dot(next.y - curr.y) + c/dimx*curr.x.dot(next.z - curr.z);
}

double AlgorithmIPFilter::CalculateSigma() const
{
    if(restoration) return params.sigma_restoration;

    if(safe_step) return (alphat < params.safe_step_threshold_alphat) ? params.sigma_safe_max : params.sigma_safe_min;

    return (curr.mu < params.mu_threshold) ? params.sigma_fast : params.sigma_slow;
}

void AlgorithmIPFilter::AcceptTrialPoint()
{
    // Update the current state curr
    curr = next;

    // Update the number of iterations
    ++iter;

    // Check if the maximum number of iterations has been achieved
    if(iter > options.max_iter)
        throw MaxIterationError();
}

void AlgorithmIPFilter::SearchDeltaNeighborhood()
{
    // Calculate the largest delta that solves the positivity conditions
    const double delta_positivity = CalculateDeltaPositivity();

    // Calculate the start trial delta so that it is not greater than the current maximum allowed
    double delta_trial = std::min(delta_positivity, delta_max);

    while(true)
    {
        // Update the members that are dependent on delta
        UpdateNextState(delta_trial);

        // Decrease the current value of the trial delta
        delta_trial *= params.delta_decrease_factor;

        // Check if delta is now less than the allowed minimum
        if(delta_trial < params.delta_min)
            throw SearchDeltaNeighborhoodError();

        // Check if the current delta results results in any IEEE floating point exception
        if(AnyFloatingPointException(next))
            continue;

        // Check if the current delta results in a point (x,y,z) that pass the centrality neighborhood condition
        if(next.thh + next.thl <= M * next.mu)
            break;
    }
}

void AlgorithmIPFilter::SearchDeltaTrustRegion()
{
    while(true)
    {
        // Calculate the linear model at the current and next states
        const double curr_m = curr.psi;
        const double next_m = CalculateNextLinearModel();

        // Check if the current trust-region radius is less than the allowed minimum
        if(delta < params.delta_min)
            throw SearchDeltaTrustRegionError();

        if(curr_m - next_m < params.kappa * curr.theta * curr.theta)
        {
            const double beta_theta = curr.theta * (1 - params.alpha_theta);
            const double beta_psi = curr.psi - params.alpha_psi * curr.theta;

            if((next.theta < beta_theta or next.psi < beta_psi) and filter.IsAcceptable({next.theta, next.psi}))
            {
                // Increase the current maximum trust-region radius
                delta_max *= params.delta_increase_factor;

                // Add a new entry to the filter
                filter.Add({beta_theta, beta_psi});

                // Reset the Lagrange multipliers z
                const double aux1 = next.mu*params.kappa_zreset;
                const double aux2 = next.mu/params.kappa_zreset;
                next.z = next.z.array().min(aux1/next.x.array()).max(aux2/next.x.array());

                // Update the neighborhood parameter M
                if(next.thh + next.thl > next.mu * params.epsilonM * M)
                    M = std::max(params.neighM_max, params.alphaM*(next.thh + next.thl)/next.mu);

                // Trial point has been found: leave loop
                break;
            }
        }
        else
        {
            // Calculate the ratio of decrease in actual and predicted optimality
            const double rho = (curr.psi - next.psi)/(curr_m - next_m);

            if(rho > params.eta_small and filter.IsAcceptable({next.theta, next.psi}))
            {
                // Increase the current maximum trust-region radius
                if(rho > params.eta_large) delta_max *= params.delta_increase_factor;

                // Reset the Lagrange multipliers z
                const double aux1 = next.mu*params.kappa_zreset;
                const double aux2 = next.mu/params.kappa_zreset;
                next.z = next.z.array().min(aux1/next.x.array()).max(aux2/next.x.array());

                // Update the neighborhood parameter M
                if(next.thh + next.thl > next.mu * params.epsilonM * M)
                    M = std::max(params.neighM_max, params.alphaM*(next.thh + next.thl)/next.mu);

                // Trial point has been found: leave loop
                break;
            }
        }

        // Decrease the trust-region radius
        UpdateNextState(params.delta_decrease_factor * delta);
    }
}

void AlgorithmIPFilter::SearchDeltaTrustRegionRestoration()
{
    // Calculate the optimality measure of the restoration algorithm at w
    const double curr_theta2 = (curr.thh*curr.thh + curr.thc*curr.thc)/2.0;

    // Calculate the d/dx and d/dz derivatives of 1/2*thh^2 and 1/2*thc^2
    const auto& ddx_thh2 = curr.h.grad.transpose() * curr.h.func;
    const auto& ddx_thc2 = (curr.z.array() * (curr.x.array() * curr.z.array() - curr.mu)).matrix();
    const auto& ddz_thc2 = (curr.x.array() * (curr.x.array() * curr.z.array() - curr.mu)).matrix();

    // Calculate the dot product of grad(theta2) with the normal step sn
    const double grad_theta2_dot_sn = (ddx_thh2 + ddx_thc2).dot(snx) + ddz_thc2.dot(snz);

    while(true)
    {
        // Calculate the optimality measure of the restoration algorithm at w(delta)
        const double next_theta2 = (next.thh*next.thh + next.thc*next.thc)/2.0;

        // Calculate the ratio of decrease in actual and predicted of theta2
        const double rho = (curr_theta2 - next_theta2)/(-grad_theta2_dot_sn);

        // Increase the current maximum trust-region radius
        if(rho > params.xi2) delta_max = delta * params.delta_increase_factor;

        // Check if rho satisfies the Cauchy condition of the restoration algorithm
        if(rho > params.xi1)
            break;

        // Decrease the current trust-region radius and update all delta dependent quantities
        UpdateNextState(delta * params.delta_decrease_factor);

        // Check if delta is now less than the allowed minimum
        if(delta < params.delta_min)
            throw SearchDeltaTrustRegionRestorationError();
    }
}

void AlgorithmIPFilter::SolveRestoration()
{
    // Add the current (theta, psi) pair to the filter
    filter.Add({curr.theta, curr.psi});

    // Activate the restoration flag
    restoration = true;

    while(true)
    {
        // Use the current normal step to find a delta that sufficiently decreases the theta2 measure
        SearchDeltaTrustRegionRestoration();

        // Check if the restoration condition and the filter acceptance condition applies
        if(PassRestorationCondition() and PassFilterCondition())
            break;

        // Accept the current trial point
        AcceptTrialPoint();

        // Calculate the new normal and tangential steps for the restoration algorithm
        UpdateNormalTangentialSteps();

        // Search for a trust-region radius that satisfies the centrality neighborhood conditions
        SearchDeltaNeighborhood();
    }

    // Deactivate the restoration flag
    restoration = false;
}

void AlgorithmIPFilter::UpdateNextState(double del)
{
    // Update the delta value
    delta = del;

    // Update the normal and tangencial step lengths
    alphan = std::min(1.0, delta/norm_sn);
    alphat = std::min(alphan, delta/norm_st);

    // Update the iterates x(delta), y(delta), and z(delta)
    next.x.noalias() = curr.x + alphan * snx + alphat * stx;
    next.y.noalias() = curr.y + alphan * sny + alphat * sty;
    next.z.noalias() = curr.z + alphan * snz + alphat * stz;

    UpdateState(next.x, next.y, next.z, next);
}

void AlgorithmIPFilter::UpdateNormalTangentialSteps()
{
    // Define some auxiliary variables
    const unsigned n = dimx;
    const unsigned m = dimy;

    // Calculate the matrix H, which is the block(1,1) of the linear system
    H = curr.f.hessian;

    for(unsigned i = 0; i < curr.h.hessian.size(); ++i)
        H += curr.y[i] * curr.h.hessian[i];

    H += curr.z.cwiseQuotient(curr.x).asDiagonal();

    // Assemble the coefficient matrix of the linear system
    lhs.block(0, 0, n, n) = H;
    lhs.block(0, n, n, m) = curr.h.grad.transpose();
    lhs.block(n, 0, m, n) = curr.h.grad;
    lhs.block(n, n, m, m) = MatrixXd::Identity(m, m);

    // Calculate the LU decomposition of the coefficient matrix
    lu.compute(lhs);

    // Assemble the normal rhs vector of the linear system
    rhs.segment(0, n) = - curr.z + (curr.mu/curr.x.array()).matrix();
    rhs.segment(n, m) = - curr.h.func;

    // Calculate the normal step
    u = lu.solve(rhs);

    // Extract the x and y components of the normal step
    snx = u.segment(0, n);
    sny = u.segment(n, m);

    // Calculate the sigma parameter
    const double sigma = CalculateSigma();

    // Assemble the tangential rhs vector of the linear system
    rhs.segment(0, n) = - curr.f.grad - curr.h.grad.transpose()*curr.y + curr.z - ((1 - sigma)*curr.mu/curr.x.array()).matrix();
    rhs.segment(n, m) = - VectorXd::Zero(m);

    // Calculate the tangential step
    u = lu.solve(rhs);

    // Extract the x and y components of the tangential step
    stx = u.segment(0, n);
    sty = u.segment(n, m);

    // Calculate the z components of the normal and tangential steps
    snz = -(curr.z.array() * snx.array() + curr.x.array() * curr.z.array() - curr.mu)/curr.x.array();
    stz = -(curr.z.array() * stx.array() + curr.mu * (1 - sigma))/curr.x.array();

    // Calculate the norms of the normal and tangential steps
    norm_sn = std::sqrt(snx.squaredNorm() + sny.squaredNorm() + snz.squaredNorm());
    norm_st = std::sqrt(stx.squaredNorm() + sty.squaredNorm() + stz.squaredNorm());
}

void AlgorithmIPFilter::UpdateState(const VectorXd& x, const VectorXd& y, const VectorXd& z, State& state) const
{
    // Initialise the iterates x, y and z
    state.x = x;
    state.y = y;
    state.z = z;

    // Initialise the objective and constraint state at x0
    state.f = problem.objective(x);
    state.h = problem.constraint(x);

    // Initialise the barrier parameter at (x,y,z)
    state.mu = x.dot(z)/dimx;

    // Initialise the auxiliary optimality theta measures at (x,y,z)
    state.thc = (x.array() * z.array() - state.mu).matrix().norm();
    state.thh = (state.h.func).norm();
    state.thl = (state.f.grad + state.h.grad.transpose() * y - z).norm();

    // Initialise the feasibility/centrality theta measure at (x,y,z)
    state.theta = state.thh + state.thc;

    // Initialise the optimality psi measure at (x,y,z)
    state.psi = (options.psi_scheme == 0) ?
        state.f.func + c * state.mu :
        state.f.func + c * state.mu + state.h.func.dot(y);
}

} /* namespace IPFilter */
} /* namespace Optima */
