/*
 * IPFilterSolver.cpp
 *
 *  Created on: 4 Apr 2013
 *      Author: allan
 */

#include "IPFilterSolver.hpp"

// C++ includes
#include <cmath>
#include <iomanip>
#include <iostream>

// Optima includes
#include <Utils/Macros.hpp>
#include <Utils/Math.hpp>
#include <IPFilter/IPFilterParams.hpp>

namespace Optima {

IPFilterSolver::IPFilterSolver()
{
}

void IPFilterSolver::SetOptions(const Options& options)
{
    this->options = options;
}

void IPFilterSolver::SetParams(const Params& params)
{
    this->params = params;
}

void IPFilterSolver::SetProblem(const OptimumProblem& problem)
{
    // Initialise the optimisation problem
    this->problem = problem;

    // Initialise the dimension variables
    dimx = problem.GetNumVariables();
    dimy = problem.GetNumConstraints();

    // Initialise the KKT linear system data
    const unsigned dim = dimx + dimy;
    lhs.resize(dim, dim);
    rhs.resize(dim);

    // Initialise the output instance
    if(options.output.active)
    {
        outputter = Outputter();
        outputter.SetOptions(options.output);
        outputter.AddEntry("iter");
        outputter.AddEntries(dimx, "x");
        outputter.AddEntry("f(x)");
        outputter.AddEntry("h(x)");
        outputter.AddEntry("error");
        outputter.AddEntry("alphan");
        outputter.AddEntry("alphat");
        outputter.AddEntry("mu(w)");
        outputter.AddEntry("delta");
        outputter.AddEntry("theta(w)");
        outputter.AddEntry("psi(w)");
        outputter.AddEntry("thc(w)");
        outputter.AddEntry("thh(w)");
        outputter.AddEntry("thl(w)");
        outputter.AddEntries(dimy, "y");
        outputter.AddEntries(dimx, "z");
    }
}

void IPFilterSolver::SetScaling(const Scaling& scaling)
{
    this->scaling = scaling;
}

const IPFilterSolver::Options& IPFilterSolver::GetOptions() const
{
    return options;
}

const IPFilterSolver::Params& IPFilterSolver::GetParams() const
{
    return params;
}

const IPFilterSolver::Result& IPFilterSolver::GetResult() const
{
    return result;
}

const IPFilterSolver::State& IPFilterSolver::GetState() const
{
    return curr;
}

const OptimumProblem& IPFilterSolver::GetProblem() const
{
    return problem;
}

void IPFilterSolver::Solve(VectorXd& x)
{
    VectorXd y, z;

    Solve(x, y, z);
}

void IPFilterSolver::Solve(VectorXd& x, VectorXd& y, VectorXd& z)
{
    // Check if the dimensions of the initial guesses x, y, and z are correct
    if(x.rows() != dimx) x.setConstant(dimx, options.xguess);
    if(y.rows() != dimy) y.setConstant(dimy, options.yguess);
    if(z.rows() != dimx) z.setConstant(dimx, options.zguess);

    // Impose the lower bound limits on the initial guesses x and z
    x = x.cwiseMax(options.xguessmin);
    z = z.cwiseMax(options.zguessmin);

    // Output the header on the top of the calculation output
    OutputHeader();

    // Initialise the result member
    result = Result();

    // Initialise the current primal variables x, and the Lagrange multipliers y and z
    curr.x.noalias() = x;
    curr.y.noalias() = y;
    curr.z.noalias() = z;

    for(unsigned attempts = 1; ; ++attempts)
    {
        // Scale the primal variables x, and the Lagrange multipliers y and z
        scaling.ScaleXYZ(curr.x, curr.y, curr.z);

        // Initialise the calculation with the provided initial guess
        Initialise(curr.x, curr.y, curr.z);

        // Solve the optimisation problem
        try { Solve(); }

        // Catch any exception that indicates a trust-region search error
        catch(const ErrorSearchDelta& e)
        {
            // Check if the number of attempts is greater than the allowed number of restart attemps
            if(attempts > params.restart.tentatives)
                { result.error = e.what(); throw; }

            // Output a message indicating the restarting algorithm
            outputter.OutputMessage("...restarting the algorithm: attempt #", attempts);

            // Unscale the primal variables x, and the Lagrange multipliers y and z
            scaling.UnscaleXYZ(curr.x, curr.y, curr.z);

            // Reset the Lagrange multipliers z
            curr.z.setConstant(dimx, std::min(0.1, std::pow(params.restart.factor, attempts) * curr.mu));

            // Reset the scaling of the primal variables x
            scaling.SetScalingVariables(curr.x);
        }

        // Catch any exception thrown in the calculation, get its error message and rethrow
        catch(const std::exception& e)
            { result.error = e.what(); throw; }

        // Check if the last calculation converged
        if(result.converged)
            break;
    }

    // Unscale the primal variables x, and the Lagrange multipliers y and z
    scaling.UnscaleXYZ(curr.x, curr.y, curr.z);

    // Transfer the found optimum point (curr.x, curr.y, curr.z) to (x, y, z)
    x.noalias() = curr.x;
    y.noalias() = curr.y;
    z.noalias() = curr.z;
}

bool IPFilterSolver::AnyFloatingPointException(const State& state) const
{
    if(isfinite(state.f.func) and isfinite(state.f.grad) and isfinite(state.f.hessian))
        if(isfinite(state.h.func) and isfinite(state.h.grad)) return false;
    return true;
}

bool IPFilterSolver::PassFilterCondition() const
{
    return filter.IsAcceptable({next.theta, next.psi});
}

bool IPFilterSolver::PassRestorationCondition(double delta) const
{
    if(not params.restoration.active)
        return true;

    const double gamma1 = params.restoration.gamma1;
    const double gamma2 = params.restoration.gamma2;
    const double beta   = params.restoration.beta;

    return curr.theta <= delta * std::min(gamma1, gamma2*std::pow(delta, beta));
}

bool IPFilterSolver::PassSafeStepCondition() const
{
    return params.safestep.active and alphat < params.safestep.threshold;
}

bool IPFilterSolver::PassConvergenceCondition() const
{
    return next.error < options.tolerance or (snx + stx).lpNorm<Infinity>() < options.tolerance;
}

double IPFilterSolver::CalculateDeltaPositiveXZ() const
{
    // The vectors x(delta) and z(delta) assuming that both alpha_n and alpha_t are 1
    const auto xcirc = curr.x + snx + stx;
    const auto zcirc = curr.z + snz + stz;

    // Check if the Trivial Case is satisfied, where alpha_n = alpha_n = 1 yields positive xcirc and zcirc
    if(xcirc.minCoeff() > 0.0 and zcirc.minCoeff() > 0.0)
        return INF;

    // Calculate the minimum and the maximum norms
    const double minnorm = std::min(norm_sn, norm_st);
    const double maxnorm = std::max(norm_sn, norm_st);

    // Calculate the auxiliary delta values for Cases I and II
    double delta_xI  = CalculateLargestBoundaryStep(curr.x, snx/norm_sn + stx/maxnorm);
    double delta_zI  = CalculateLargestBoundaryStep(curr.z, snz/norm_sn + stz/maxnorm);
    double delta_xII = CalculateLargestBoundaryStep(curr.x + snx, stx/norm_st);
    double delta_zII = CalculateLargestBoundaryStep(curr.z + snz, stz/norm_st);

    // Return the minimum among all other delta values
    return 0.99 * std::min({delta_xI, delta_zI, delta_xII, delta_zII});
}

double IPFilterSolver::CalculateDeltaXzGreaterGammaMu() const
{
    // Check if the neighbourhood search algorithm is not activated
    if(not params.neighbourhood.active)
        return INF;

    // Define some auxiliary variables
    const auto& ones = VectorXd::Ones(dimx);

    // The vectors x(delta) and z(delta) assuming that both alpha_n and alpha_t are 1
    const VectorXd xcirc = curr.x + snx + stx;
    const VectorXd zcirc = curr.z + snz + stz;

    // Check if the Trivial Case is satisfied, where alpha_n = alpha_n = 1 yields X(delta)z(delta) > gamma*mu(delta)
    if((xcirc.cwiseProduct(zcirc) - gamma/dimx * xcirc.dot(zcirc) * ones).minCoeff() > 0.0)
        return INF;

    // Calculate the the maximum among the two norms norm_sn and norm_st
    const double maxnorm = std::max(norm_sn, norm_st);

    // Calculate the auxiliary delta values for Cases I and II
    const double delta_xzI  = CalculateLargestQuadraticStep(curr.x + snx, stx/norm_st, curr.z + snz, stz/norm_st);
    const double delta_xzII = CalculateLargestQuadraticStep(curr.x, snx/norm_sn + stx/maxnorm, curr.z, snz/norm_sn + stz/maxnorm);

    // Calculate the minimum among all previous delta values
    return std::min(delta_xzI, delta_xzII);
}

double IPFilterSolver::CalculateLargestBoundaryStep(const VectorXd& p, const VectorXd& dp) const
{
    double step = INF;
    for(unsigned i = 0; i < dimx; ++i)
    {
        const double aux = -p[i]/dp[i];
        if(aux > 0.0) step = std::min(step, aux);
    }
    return step;
}

double IPFilterSolver::CalculateLargestQuadraticStep(
    const VectorXd& a, const VectorXd& b, const VectorXd& c, const VectorXd& d) const
{
    const VectorXd a1 = b.array() * d.array() * dimx - b.dot(d) * gamma;
    const VectorXd a2 = (a.array() * d.array() + b.array() * c.array()) * dimx - (a.dot(d) + b.dot(c)) * gamma;
    const VectorXd a3 = a.array() * c.array() * dimx - a.dot(c) * gamma;

    auto solve = [](double a, double b, double c) -> double
    {
        const double aux = std::sqrt(b*b - 4*a*c);

        const double r1 = (-b + aux)/(2*a);
        const double r2 = (-b - aux)/(2*a);

        return std::min(positive(r1), positive(r2));
    };

    double step = solve(a1[0], a2[0], a3[0]);
    for(unsigned i = 1; i < dimx; ++i)
        step = std::min(step, solve(a1[i], a2[i], a3[i]));

    return step;
}

double IPFilterSolver::CalculateNextLinearModel() const
{
    VectorXd psix = VectorXd::Zero(dimx);
    VectorXd psiy = VectorXd::Zero(dimy);
    VectorXd psiz = VectorXd::Zero(dimx);

    switch(options.psi)
    {
    case PsiObjective:
        psix.noalias() = curr.f.grad + c/dimx*curr.z;
        psiz.noalias() = c/dimx*curr.x;
        break;
    case PsiLagrange:
        psix.noalias() = curr.f.grad + curr.h.grad.transpose()*curr.y + c/dimx*curr.z;
        psiy.noalias() = curr.h.func;
        psiz.noalias() = c/dimx*curr.x;
        break;
    case PsiGradLagrange:
        psix.noalias() = 2*Lxx.transpose()*Lx + curr.z/dimx;
        psiy.noalias() = 2*curr.h.grad*Lx;
        psiz.noalias() =-2*Lx + curr.x/dimx;
        break;
    }

    return curr.psi + psix.dot(next.x - curr.x) + psiy.dot(next.y - curr.y) + psiz.dot(next.z - curr.z);
}

double IPFilterSolver::CalculatePsi(const State& state) const
{
    switch(options.psi)
    {
    case PsiObjective:
        return state.f.func + c * state.mu;
    case PsiLagrange:
        return state.f.func + c * state.mu + state.h.func.dot(state.y);
    case PsiGradLagrange:
        return (state.f.grad + state.h.grad.transpose()*state.y - state.z).squaredNorm() + state.mu;
    }
}

double IPFilterSolver::CalculateSigma() const
{
    if(restoration) return params.sigma.restoration;

    switch(options.sigma)
    {
    case SigmaDefault:
        return CalculateSigmaDefault();
    case SigmaLOQO:
        return CalculateSigmaLOQO();
    }
}

double IPFilterSolver::CalculateSigmaDefault() const
{
    return (curr.mu < params.sigma.threshold_mu) ? params.sigma.main_max : params.sigma.main_min;
}

double IPFilterSolver::CalculateSigmaLOQO() const
{
    const double xi = curr.x.cwiseProduct(curr.z).minCoeff()/curr.mu;

    return 0.1 * std::pow(std::min(0.05*(1 - xi)/xi, 2.0), 3);
}

void IPFilterSolver::AcceptTrialPoint()
{
    // Update the current state curr
    curr = next;

    // Update the number of iterations
    ++result.num_iterations;

    // Check if the maximum number of iterations has been achieved
    if(result.num_iterations > options.max_iterations)
        throw ErrorIterationMaximumLimit();
}

void IPFilterSolver::ExtendFilter()
{
    // Calculate the components of the new entry
    const double beta_theta = curr.theta * (1 - params.filter.alpha_theta);
    const double beta_psi = curr.psi - params.filter.alpha_psi * curr.theta;

    // Add a new entry to the filter
    filter.Add({beta_theta, beta_psi});
}

void IPFilterSolver::Initialise(const VectorXd& x, const VectorXd& y, const VectorXd& z)
{
    // Initialise the filter
    filter = Filter();

    // Initialise the value of the parameter gamma
    gamma = std::min(params.neighbourhood.gamma_min, x.cwiseProduct(z).minCoeff()/(2.0*x.dot(z)/dimx));

    // Initialise the value of the parameter c
    c = 3*dimx*dimx/(1 - params.sigma.main_min)*std::pow(std::max(1.0, (1 - params.sigma.main_min)/gamma), 2);

    // Update the current state
    UpdateState(x, y, z, curr);

    // Initialise the value of the neighbourhood parameter M
    M = std::max(params.neighbourhood.Mmax, params.neighbourhood.alpha0*(curr.thh + curr.thl)/curr.mu);

    // Check if the initial guess results in floating point exceptions
    if(AnyFloatingPointException(curr))
        throw ErrorInitialGuessFloatingPoint();

    // Initialise the current maximum value of the trust-region radius
    delta = delta_initial = params.main.delta_initial;

    // Initialise the normal and tangencial step-lengths respectively
    alphan = alphat = 1.0;

    // Initialise the boolean values to false
    restoration = false;

    // Initialise the next state with the just initialised current state
    next = curr;
}

void IPFilterSolver::OutputHeader()
{
    if(options.output.active) outputter.OutputHeader();
}

void IPFilterSolver::OutputState()
{
    if(options.output.active)
    {
        if(not options.output_scaled)
        {
            scaling.UnscaleX(curr.x);
            scaling.UnscaleY(curr.y);
            scaling.UnscaleZ(curr.z);
        }

        outputter.AddValue(result.num_iterations);
        outputter.AddValues(curr.x.data(), curr.x.data() + dimx);
        outputter.AddValue(curr.f.func);
        outputter.AddValue(curr.h.func.norm());
        outputter.AddValue(curr.error);
        outputter.AddValue(alphan);
        outputter.AddValue(alphat);
        outputter.AddValue(curr.mu);
        outputter.AddValue(delta);
        outputter.AddValue(curr.theta);
        outputter.AddValue(curr.psi);
        outputter.AddValue(curr.thc);
        outputter.AddValue(curr.thh);
        outputter.AddValue(curr.thl);
        outputter.AddValues(curr.y.data(), curr.y.data() + dimy);
        outputter.AddValues(curr.z.data(), curr.z.data() + dimx);
        outputter.OutputState();

        if(not options.output_scaled)
        {
            scaling.ScaleX(curr.x);
            scaling.ScaleY(curr.y);
            scaling.ScaleZ(curr.z);
        }
    }
}

void IPFilterSolver::SearchDeltaNeighborhood()
{
    // Calculate the delta sizes that yield x,z > 0 and Xz > gamma*mu
    const double delta1 = CalculateDeltaPositiveXZ();
    const double delta2 = CalculateDeltaXzGreaterGammaMu();

    // Calculate the adjustment factor tau
    const double tau = 1.0 - std::min(0.01, 100.0 * curr.mu * curr.mu);

    // Calculate the damped delta size that yield simultaneously yield x,z > 0 and Xz > gamma*mu
    const double delta_max = tau*std::min(delta1, delta2);

    // Calculate the start trial delta so that it is not greater than the current maximum allowed
    double trial = std::min(delta_max, delta_initial);

    // Begin the neighbourhood search loop
    while(true)
    {
        // Update the next state with the new delta value
        UpdateNextState(trial);

        // Decrease the current value of the trial delta
        trial *= params.main.delta_decrease;

        // Check if delta is now less than the allowed minimum
        if(trial < params.main.delta_min)
            throw ErrorSearchDeltaNeighborhood();

        // Check if the current delta results results in any IEEE floating point exception
        if(AnyFloatingPointException(next))
            continue;

        // Prevent further tentative delta sizes if the neighbourhood search algorithm deactivated
        if(not params.neighbourhood.active)
            break;

        // Check if the current delta results in a point (x,y,z) that pass the centrality neighborhood condition
        if(next.thh + next.thl <= M * next.mu)
            break;
    }
}

void IPFilterSolver::SearchDeltaTrustRegion()
{
    while(true)
    {
        // Check if the current trust-region radius is less than the allowed minimum
        if(delta < params.main.delta_min)
            throw ErrorSearchDeltaTrustRegion();

        // Calculate the linear model at the current and next states
        const double curr_m = curr.psi;
        const double next_m = CalculateNextLinearModel();

        // Calculate the ratio of decrease in actual and predicted optimality
        const double rho = (curr.psi - next.psi)/(curr_m - next_m);

        // Check if w(delta) pass the filter condition
        if(PassFilterCondition())
        {
            if(curr_m - next_m < params.main.kappa*curr.theta*curr.theta)
            {
                // Extend the filter with the current (theta, psi) pair
                ExtendFilter();

                // Update the current maximum trust-region radius
                delta_initial = delta * params.main.delta_increase;

                // Trial point has been found: leave loop
                break;
            }

            if(rho > params.main.eta1)
            {
                // Update the current maximum trust-region radius
                delta_initial = (rho > params.main.eta2) ?
                    delta * params.main.delta_increase : delta;

                // Trial point has been found: leave loop
                break;
            }
        }

        // Decrease the trust-region radius
        UpdateNextState(params.main.delta_decrease * delta);
    }

    // Update the neighborhood parameter M
    UpdateNeighborhoodParameterM();

    // Accept the found delta and update the current state
    AcceptTrialPoint();
}

void IPFilterSolver::SearchDeltaTrustRegionRestoration()
{
    // Calculate the optimality measure of the restoration algorithm at w
    const double curr_theta2 = (curr.thh*curr.thh + curr.thc*curr.thc)/2.0;

    while(true)
    {
        // Calculate the optimality measure of the restoration algorithm at w(delta)
        const double next_theta2 = (next.thh*next.thh + next.thc*next.thc)/2.0;

        // Calculate the ratio of decrease in actual and predicted of theta2
        const double rho = (curr_theta2 - next_theta2)/(2.0*curr_theta2);

        // Increase the current maximum trust-region radius
        if(rho > params.restoration.xi2) delta_initial = delta * params.main.delta_increase;

        // Check if rho satisfies the Cauchy condition of the restoration algorithm
        if(rho > params.restoration.xi1)
            break;

        // Decrease the current trust-region radius and update all delta dependent quantities
        UpdateNextState(delta * params.main.delta_decrease);

        // Check if delta is now less than the allowed minimum
        if(delta < params.main.delta_min)
            throw ErrorSearchDeltaTrustRegionRestoration();
    }
}

void IPFilterSolver::Solve()
{
    while(true)
    {
        // Output the current state of the calculation
        OutputState();

        // Calculate the normal and tangential steps for the trust-region algorithm
        UpdateNormalTangentialSteps();

        // Search for a trust-region radius that satisfies the centrality neighborhood conditions
        SearchDeltaNeighborhood();

        // Check if the current tangential step needs to be safely recalculated
        if(PassSafeStepCondition())
        {
            // Calculate the safe tangential step
            UpdateSafeTangentialStep();

            // Repeat the search for a trust-region radius that satisfies the neighborhood conditions
            SearchDeltaNeighborhood();
        }

        // Check if the current state does not require the initiation of the restoration phase algorithm
        if(PassRestorationCondition(delta)) SearchDeltaTrustRegion();

        // Start the restoration algorithm in order to decrease the primal infeasibility
        else SolveRestoration();

        // Check if the current state pass the stopping criteria of optimality
        if(PassConvergenceCondition()) break;
    }

    // Output the final state of the calculation
    OutputState();

    // Update the convergency condition of result
    result.converged = true;
}

void IPFilterSolver::SolveRestoration()
{
    // Extend the filter with the current (theta, psi) pair
    ExtendFilter();

    // Activate the restoration flag
    restoration = true;

    // Update the restoration counter of the result
    ++result.num_restorations;

    // Output a message indicating the start of the restoration algorithm
    outputter.OutputMessage("...beginning the restoration algorithm");

    // Store the current value of the trust-region radius that was a result of the main iterations so far
    double delta_main = delta;

    while(true)
    {
        // Calculate the new normal and tangential steps for the restoration algorithm
        UpdateNormalTangentialStepsRestoration();

        // Search for a trust-region radius that satisfies the centrality neighborhood conditions
        SearchDeltaNeighborhood();

        // Use the current normal step to find a delta that sufficiently decreases the theta2 measure
        SearchDeltaTrustRegionRestoration();

        // Accept the current trial point
        AcceptTrialPoint();

        // Output the current state of the calculation in the restoration algorithm
        OutputState();

        // Check if the restoration condition pass and the filter acceptance condition applies
        if(PassRestorationCondition(delta_main) and PassFilterCondition())
            break;
    }

    // Recover the value of delta
    delta = delta_initial = delta_main;

    // Output a message indicating the end of the restoration algorithm
    outputter.OutputMessage("...finishing the restoration algorithm");

    // Deactivate the restoration flag
    restoration = false;
}

void IPFilterSolver::UpdateNeighborhoodParameterM()
{
    if(next.theta > next.mu * params.neighbourhood.epsilon * M)
        M = std::max(params.neighbourhood.Mmax, params.neighbourhood.alpha*next.theta/next.mu);
}

void IPFilterSolver::UpdateNextState(double del)
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

void IPFilterSolver::UpdateNormalTangentialSteps()
{
    // Define some auxiliary variables
    const unsigned n = dimx;
    const unsigned m = dimy;

    // Calculate the gradient of the Lagrange function with respect to x at the current state
    Lx.noalias() = curr.f.grad + curr.h.grad.transpose()*curr.y - curr.z;

    // Calculate the Hessian of the Lagrange function with respect to x at the current state
    Lxx = curr.f.hessian;
    for(unsigned i = 0; i < curr.h.hessian.size(); ++i)
        Lxx += curr.y[i] * curr.h.hessian[i];

    // Assemble the coefficient matrix of the linear system
    lhs.block(0, 0, n, n).noalias() = Lxx;
    lhs.block(0, 0, n, n) += curr.z.cwiseQuotient(curr.x).asDiagonal();
    lhs.block(0, n, n, m).noalias() = curr.h.grad.transpose();
    lhs.block(n, 0, m, n).noalias() = curr.h.grad;
    lhs.block(n, n, m, m).noalias() = MatrixXd::Zero(m, m);

    // Calculate the LU decomposition of the coefficient matrix
    lu.compute(lhs);

    // Assemble the normal rhs vector of the linear system
    rhs.segment(0, n).noalias() = - curr.z + (curr.mu/curr.x.array()).matrix();
    rhs.segment(n, m).noalias() = - curr.h.func;

    // Calculate the normal step
    u = lu.solve(rhs);

    // Extract the x and y components of the normal step
    snx.noalias() = u.segment(0, n);
    sny.noalias() = u.segment(n, m);

    // Calculate the sigma parameter
    const double sigma = CalculateSigma();

    // Assemble the tangential rhs vector of the linear system
    rhs.segment(0, n).noalias() = - Lx - ((1 - sigma)*curr.mu/curr.x.array()).matrix();
    rhs.segment(n, m).noalias() = VectorXd::Zero(m);

    // Calculate the tangential step
    u = lu.solve(rhs);

    // Extract the x and y components of the tangential step
    stx.noalias() = u.segment(0, n);
    sty.noalias() = u.segment(n, m);

    // Calculate the z components of the normal and tangential steps
    snz = -(curr.z.array() * snx.array() + curr.x.array() * curr.z.array() - curr.mu)/curr.x.array();
    stz = -(curr.z.array() * stx.array() + curr.mu * (1 - sigma))/curr.x.array();

    // Calculate the norms of the normal and tangential steps
    norm_sn = std::sqrt(snx.squaredNorm() + sny.squaredNorm() + snz.squaredNorm());
    norm_st = std::sqrt(stx.squaredNorm() + sty.squaredNorm() + stz.squaredNorm());
}

void IPFilterSolver::UpdateNormalTangentialStepsRestoration()
{
    // Define some auxiliary variables
    const unsigned n = dimx;
    const unsigned m = dimy;

    // Calculate the gradient of the Lagrange function with respect to x at the current state
    Lx.noalias() = curr.f.grad + curr.h.grad.transpose()*curr.y - curr.z;

    // Calculate the Hessian of the Lagrange function with respect to x at the current state
    Lxx = curr.f.hessian;
    for(unsigned i = 0; i < curr.h.hessian.size(); ++i)
        Lxx += curr.y[i] * curr.h.hessian[i];

    // Assemble the coefficient matrix of the linear system
    lhs.block(0, 0, n, n).noalias() = Lxx;
    lhs.block(0, 0, n, n) += curr.z.cwiseQuotient(curr.x).asDiagonal();
    lhs.block(0, n, n, m).noalias() = curr.h.grad.transpose();
    lhs.block(n, 0, m, n).noalias() = curr.h.grad;
    lhs.block(n, n, m, m).noalias() = -curr.mu * MatrixXd::Identity(m, m);

    // Calculate the LU decomposition of the coefficient matrix
    lu.compute(lhs);

    // Assemble the normal rhs vector of the linear system
    rhs.segment(0, n).noalias() = - curr.z + (curr.mu/curr.x.array()).matrix();
    rhs.segment(n, m).noalias() = - curr.h.func;

    // Calculate the normal step
    u = lu.solve(rhs);

    // Extract the x and y components of the normal step
    snx.noalias() = u.segment(0, n);
    sny.noalias() = u.segment(n, m);

    // Calculate the sigma parameter
    const double sigma = CalculateSigma();

    // Assemble the tangential rhs vector of the linear system
    rhs.segment(0, n).noalias() = - ((1 - sigma)*curr.mu/curr.x.array()).matrix();
    rhs.segment(n, m).noalias() = VectorXd::Zero(m);

    // Calculate the tangential step
    u = lu.solve(rhs);

    // Extract the x and y components of the tangential step
    stx.noalias() = u.segment(0, n);
    sty.noalias() = u.segment(n, m);

    // Calculate the z components of the normal and tangential steps
    snz = -(curr.z.array() * snx.array() + curr.x.array() * curr.z.array() - curr.mu)/curr.x.array();
    stz = -(curr.z.array() * stx.array() + curr.mu * (1 - sigma))/curr.x.array();

    // Calculate the norms of the normal and tangential steps
    norm_sn = std::sqrt(snx.squaredNorm() + sny.squaredNorm() + snz.squaredNorm());
    norm_st = std::sqrt(stx.squaredNorm() + sty.squaredNorm() + stz.squaredNorm());
}

void IPFilterSolver::UpdateSafeTangentialStep()
{
    // Define some auxiliary variables
    const unsigned n = dimx;
    const unsigned m = dimy;

    // Calculate the value of parameter sigma for the safe tangential step re-calculation
    const double sigma = (alphat < params.sigma.threshold_alphat) ? params.sigma.safe_max : params.sigma.safe_min;

    // Assemble the tangential rhs vector of the linear system
    rhs.segment(0, n).noalias() = - Lx - ((1 - sigma)*curr.mu/curr.x.array()).matrix();
    rhs.segment(n, m).noalias() = VectorXd::Zero(m);

    // Calculate the safe tangential step
    u = lu.solve(rhs);

    // Extract the x and y components of the safe tangential step
    stx.noalias() = u.segment(0, n);
    sty.noalias() = u.segment(n, m);

    // Calculate the z components of the safe tangential step
    stz = -(curr.z.array() * stx.array() + curr.mu * (1 - sigma))/curr.x.array();

    // Calculate the norm of the safe tangential steps
    norm_st = std::sqrt(stx.squaredNorm() + sty.squaredNorm() + stz.squaredNorm());
}

void IPFilterSolver::UpdateState(const VectorXd& x, const VectorXd& y, const VectorXd& z, State& state)
{
    // Update the iterates x, y and z
    state.x.noalias() = x;
    state.y.noalias() = y;
    state.z.noalias() = z;

    // Update the objective and constraint states at x
    scaling.UnscaleX(state.x);
        state.f = problem.Objective(state.x);
        state.h = problem.Constraint(state.x);
    scaling.ScaleX(state.x);

    // Scale the objective and constraint states at x
    scaling.ScaleObjective(state.f);
    scaling.ScaleConstraint(state.h);

    // Update the counter of objective and constraint evaluations in result
    ++result.num_objective_evals;
    ++result.num_constraint_evals;

    // Update the barrier parameter at (x,y,z)
    state.mu = x.dot(z)/dimx;

    // Update the auxiliary optimality theta measures at (x,y,z)
    state.thc = (x.array() * z.array() - state.mu).matrix().norm();
    state.thh = (state.h.func).norm();
    state.thl = (state.f.grad + state.h.grad.transpose()*y - z).norm();

    // Update the feasibility/centrality theta measure at (x,y,z)
    state.theta = state.thh + state.thc;

    // Update the optimality psi measure at (x,y,z)
    state.psi = CalculatePsi(state);

    // Update feasibility, centrality, and optimality errors
    const double sc = 0.01 * std::max(100.0, state.z.lpNorm<1>()/dimx);
    const double sl = 0.01 * std::max(100.0, (state.y.lpNorm<1>() + state.z.lpNorm<1>())/(dimx + dimy));

    state.errorh = state.h.func.lpNorm<Infinity>();
    state.errorc = 1.0/sc * x.cwiseProduct(z).lpNorm<Infinity>();
    state.errorl = 1.0/sl * (state.f.grad + state.h.grad.transpose()*y - z).lpNorm<Infinity>();

    // Update the maximum among the feasibility, centrality, and optimality errors
    state.error = std::max({state.errorh, state.errorc, state.errorl});
}

} /* namespace Optima */
