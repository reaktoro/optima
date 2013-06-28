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
#include <Optima/Utils/Macros.hpp>
#include <Optima/Utils/Math.hpp>
#include <Optima/IPFilter/IPFilterParams.hpp>

namespace Optima {

IPFilterSolver::IPFilterSolver()
: dimx(0), dimy(0)
{}

void IPFilterSolver::SetOptions(const IPFilterOptions& options_)
{
    options = options_;

    // Initialise the outputter instance (because the options.output might have been changed)
    InitialiseOutputter();
}

void IPFilterSolver::SetParams(const IPFilterParams& params_)
{
    params = params_;
}

void IPFilterSolver::SetProblem(const OptimumProblem& problem_)
{
    // Initialise the optimisation problem
    problem = problem_;

    // Initialise the dimension variables
    dimx = problem.GetNumVariables();
    dimy = problem.GetNumConstraints();

    // Initialise the KKT linear system data
    lhs.resize(dimx + dimy, dimx + dimy);
    rhs.resize(dimx + dimy);

    // Initialise the outputter instance (because dimx and dimy might have been changed)
    InitialiseOutputter();
}

void IPFilterSolver::SetScaling(const Scaling& scaling_)
{
    scaling = scaling_;
}

const IPFilterOptions& IPFilterSolver::GetOptions() const
{
    return options;
}

const IPFilterParams& IPFilterSolver::GetParams() const
{
    return params;
}

const IPFilterResult& IPFilterSolver::GetResult() const
{
    return result;
}

const IPFilterState& IPFilterSolver::GetState() const
{
    return curr;
}

const OptimumProblem& IPFilterSolver::GetProblem() const
{
    return problem;
}

bool IPFilterSolver::Converged() const
{
    return next.error < options.tolerance1 or (snx + stx).lpNorm<Infinity>() < options.tolerance2;
}

void IPFilterSolver::Initialise(VectorXd& x, VectorXd& y, VectorXd& z)
{
    // Check if the dimensions of the initial guesses x, y, and z are correct
    if(x.rows() != dimx) x.setConstant(dimx, options.initialguess.x);
    if(y.rows() != dimy) y.setConstant(dimy, options.initialguess.y);
    if(z.rows() != dimx) z.setConstant(dimx, options.initialguess.z);

    // Impose the lower bound limits on the initial guesses x and z
    x = x.cwiseMax(options.initialguess.xmin);
    z = z.cwiseMax(options.initialguess.zmin);

    // Output the header of the calculation
    OutputHeader();

    // Initialise the result member
    result = IPFilterResult();

    // Initialise the remaining of the internal state
    InitialiseAuxiliary(x, y, z);
}

void IPFilterSolver::Iterate(VectorXd& x, VectorXd& y, VectorXd& z)
{
    // Check if the Newton algorithm is to be applied
    if(params.newton.active and curr.mu < params.newton.threshold)
        IterateNewton(x, y, z);

    // Apply the trust-region algorithm
    else IterateTrustRegion(x, y, z);

    // Transfer the current iterate state to (x, y, z)
    x.noalias() = curr.x;
    y.noalias() = curr.y;
    z.noalias() = curr.z;
}

void IPFilterSolver::Solve(VectorXd& x)
{
    // Create auxiliary Lagrange multipliers
    VectorXd y, z;

    // Solve the optimisation problem
    Solve(x, y, z);
}

void IPFilterSolver::Solve(VectorXd& x, VectorXd& y, VectorXd& z)
{
    // Initialise the optimisation solver
    Initialise(x, y, z);

    // Iterate until convergence
    do { Iterate(x, y, z); } while(not Converged());

    // Output the found optimum state
    OutputState();

    // Unscale the iterates (x, y, z)
    scaling.UnscaleXYZ(x, y, z);

    // Set the converged flag in the result member
    result.converged = true;
}

bool IPFilterSolver::AnyFloatingPointException(const IPFilterState& state) const
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
    return params.safe_step.active and alphat < params.safe_step.threshold;
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

    switch(params.psi.scheme)
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

double IPFilterSolver::CalculatePsi(const IPFilterState& state) const
{
    switch(params.psi.scheme)
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
    switch(params.sigma.scheme)
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

double IPFilterSolver::CalculateSigmaQuality()
{
    Assert(false, "IPFilterSolver::CalculateSigmaQuality() seems not to work with the ipfilter algorithm.");

    // Define some auxiliary variables
    const unsigned n = dimx;
    const unsigned m = dimy;

    // Define some auxiliary references
    auto& data = quality.GetData();
    auto& dx0  = data.dx0;
    auto& dy0  = data.dy0;
    auto& dz0  = data.dz0;
    auto& dx1  = data.dx1;
    auto& dy1  = data.dy1;
    auto& dz1  = data.dz1;

    // Assemble the rhs vector of the linear system using sigma = 0
    rhs.segment(0, n).noalias() = - (Lx + curr.z);
    rhs.segment(n, m).noalias() = - curr.h.func;

    // Calculate the direction u = [dx0, dy0]
    u = lu.solve(rhs);

    // Extract the dx0 and dy0 components of u = [dx0, dy0]
    dx0.noalias() = u.segment(0, n);
    dy0.noalias() = u.segment(n, m);

    // Calculate the direction dz0 = dz(0)
    dz0 = -(curr.z.array() * dx0.array() + curr.x.array()*curr.z.array())/curr.x.array();

    // Assemble the rhs vector of the linear system using sigma = 1
    rhs.segment(0, n).noalias() = - (Lx + curr.z - (curr.mu/curr.x.array()).matrix());
    rhs.segment(n, m).noalias() = - curr.h.func;

    // Calculate the direction u = [dx1, dy1]
    u = lu.solve(rhs);

    // Extract the dx1 and dy1 components of u = [dx1, dy1]
    dx1.noalias() = u.segment(0, n);
    dy1.noalias() = u.segment(n, m);

    // Calculate the direction dz1 = dz(1)
    dz1 = -(curr.z.array() * dx1.array() + curr.x.array()*curr.z.array() - curr.mu)/curr.x.array();

    data.x.noalias() = curr.x;
    data.y.noalias() = curr.y;
    data.z.noalias() = curr.z;

    data.thh = curr.thh;
    data.thl = curr.thl;

    quality.SetData(data);

    return quality.CalculateSigma();
}

double IPFilterSolver::CalculateSigmaRestoration() const
{
    return params.sigma.restoration;
}

double IPFilterSolver::CalculateSigmaSafeStep() const
{
    return (alphat < params.sigma.threshold_alphat) ? params.sigma.safe_min : params.sigma.safe_max;
}

void IPFilterSolver::AcceptTrialPoint()
{
    // Update the number of iterations
    ++result.num_iterations;

    // Check if the maximum number of iterations has been achieved
    if(result.num_iterations > options.max_iterations)
        throw IPFilterErrorIterationAttemptLimit();

    // Update the previous and current states
    prev = curr;
    curr = next;
}

void IPFilterSolver::ExtendFilter()
{
    // Calculate the components of the new entry
    const double beta_theta = curr.theta * (1 - params.filter.alpha_theta);
    const double beta_psi = curr.psi - params.filter.alpha_psi * curr.theta;

    // Add a new entry to the filter
    filter.Add({beta_theta, beta_psi});
}

void IPFilterSolver::InitialiseAuxiliary(VectorXd& x, VectorXd& y, VectorXd& z)
{
    // Scale the iterates (x, y, z)
    scaling.ScaleXYZ(x, y, z);

    // Initialise the current primal variables x, and the Lagrange multipliers y and z
    curr.x.noalias() = x;
    curr.y.noalias() = y;
    curr.z.noalias() = z;

    // Initialise the filter
    filter = Filter();

    // Initialise the value of the parameter gamma
    gamma = curr.x.cwiseProduct(curr.z).minCoeff()/(2.0*curr.x.dot(curr.z)/dimx);
    gamma = std::min(gamma, params.neighbourhood.gamma_min);

    // Initialise the value of the parameter c
    c = 3*dimx*dimx/(1 - params.sigma.main_min)*std::pow(std::max(1.0, (1 - params.sigma.main_min)/gamma), 2);

    // Update the current state
    UpdateState(curr.x, curr.y, curr.z, curr);

    // Check if the initial guess results in floating point exceptions
    if(AnyFloatingPointException(curr))
        throw IPFilterErrorInitialGuessFloatingPoint();

    // Initialise the value of the neighbourhood parameter M
    M = std::max(params.neighbourhood.Mmax, params.neighbourhood.alpha0*(curr.thh + curr.thl)/curr.mu);

    // Initialise the current maximum value of the trust-region radius
    delta = delta_initial = params.trust_region.delta_initial;

    // Initialise the normal and tangencial step-lengths respectively
    alphan = alphat = 1.0;

    // Initialise the next state with the just initialised current state
    prev = next = curr;
}

void IPFilterSolver::InitialiseOutputter()
{
    // Reset the outputter instance
    outputter = Outputter();

    // Set the outputter options
    outputter.SetOptions(options.output);

    // Initialise the outputter instance
    if(options.output.active)
    {
        if(options.output.iter)     outputter.AddEntry("iter");
        if(options.output.x)        outputter.AddEntries(dimx, "x");
        if(options.output.y)        outputter.AddEntries(dimy, "y");
        if(options.output.z)        outputter.AddEntries(dimx, "z");
        if(options.output.f)        outputter.AddEntry("f(x)");
        if(options.output.h)        outputter.AddEntry("h(x)");
        if(options.output.mu)       outputter.AddEntry("mu(w)");
        if(options.output.error)    outputter.AddEntry("error");
        if(options.output.residual) outputter.AddEntry("residual");
        if(options.output.alphan)   outputter.AddEntry("alphan");
        if(options.output.alphat)   outputter.AddEntry("alphat");
        if(options.output.delta)    outputter.AddEntry("delta");
        if(options.output.theta)    outputter.AddEntry("theta(w)");
        if(options.output.psi)      outputter.AddEntry("psi(w)");
        if(options.output.thc)      outputter.AddEntry("thc(w)");
        if(options.output.thh)      outputter.AddEntry("thh(w)");
        if(options.output.thl)      outputter.AddEntry("thl(w)");
    }
}

void IPFilterSolver::IterateNewton(VectorXd& x, VectorXd& y, VectorXd& z)
{
    // Output the current state of the calculation
    OutputState();

    // Calculate the x, y, and z steps for the Newton algorithm
    UpdateNewtonSteps();

    // Update the next state with the x, y, and z Newton steps
    UpdateNewtonNextState();

    // Accept the new iterates
    AcceptTrialPoint();
}

void IPFilterSolver::IterateTrustRegion(VectorXd& x, VectorXd& y, VectorXd& z)
{
    // Output the current state of the calculation
    OutputState();

    // Calculate the normal and tangential steps for the trust-region algorithm
    UpdateTrustRegionSteps();

    // Keep track if any trust-region search error happens from now on
    try
    {
        // Search for a trust-region radius that satisfies the centrality neighborhood conditions
        SearchDeltaNeighborhood();

        // Check if the current tangential step needs to be safely recalculated
        if(PassSafeStepCondition())
        {
            // Calculate the safe tangential step
            UpdateTrustRegionSafeTangentialStep();

            // Repeat the search for a trust-region radius that satisfies the neighborhood conditions
            SearchDeltaNeighborhood();
        }

        // Check if the current state does not require the initiation of the restoration phase algorithm
        if(PassRestorationCondition(delta)) SearchDeltaTrustRegion();

        // Start the restoration algorithm in order to decrease the primal infeasibility
        else SolveRestoration();
    }

    // Catch any exception that indicates a trust-region search error
    catch(const IPFilterErrorSearchDelta& e)
    {
        // Perform the restart procedure to continue the calculation with an increased perturbation
        Restart();
    }
}

void IPFilterSolver::OutputHeader()
{
    outputter.OutputHeader();
}

void IPFilterSolver::OutputState()
{
    if(options.output.active)
    {
        if(not options.output.scaled)
        {
            scaling.UnscaleX(curr.x);
            scaling.UnscaleY(curr.y);
            scaling.UnscaleZ(curr.z);
        }

        if(options.output.iter)     outputter.AddValue(result.num_iterations);
        if(options.output.x)        outputter.AddValues(curr.x.data(), curr.x.data() + dimx);
        if(options.output.y)        outputter.AddValues(curr.y.data(), curr.y.data() + dimy);
        if(options.output.z)        outputter.AddValues(curr.z.data(), curr.z.data() + dimx);
        if(options.output.f)        outputter.AddValue(curr.f.func);
        if(options.output.h)        outputter.AddValue(curr.h.func.norm());
        if(options.output.mu)       outputter.AddValue(curr.mu);
        if(options.output.error)    outputter.AddValue(curr.error);
        if(options.output.residual) outputter.AddValue(curr.residual);
        if(options.output.alphan)   outputter.AddValue(alphan);
        if(options.output.alphat)   outputter.AddValue(alphat);
        if(options.output.delta)    outputter.AddValue(delta);
        if(options.output.theta)    outputter.AddValue(curr.theta);
        if(options.output.psi)      outputter.AddValue(curr.psi);
        if(options.output.thc)      outputter.AddValue(curr.thc);
        if(options.output.thh)      outputter.AddValue(curr.thh);
        if(options.output.thl)      outputter.AddValue(curr.thl);

        outputter.OutputState();

        if(not options.output.scaled)
        {
            scaling.ScaleX(curr.x);
            scaling.ScaleY(curr.y);
            scaling.ScaleZ(curr.z);
        }
    }
}

void IPFilterSolver::Restart()
{
    // Update the number of times the calculation entered the restart algorithm
    ++result.num_restarts;

    // Output a message indicating the restarting algorithm
    outputter.OutputMessage("...restarting the algorithm at attempt number: ", result.num_restarts);

    // Check if the number of attempts is greater than the allowed number of restart attemps
    if(result.num_restarts > params.restart.tentatives)
        throw IPFilterErrorRestartAttemptLimit();

    // Unscale the iterates (x, y, z)
    scaling.UnscaleXYZ(curr.x, curr.y, curr.z);

    // Reset the Lagrange multipliers z
    curr.z.fill(std::min(options.initialguess.z, std::pow(params.restart.factor, result.num_restarts) * curr.mu));

    // Reset the scaling of the primal variables x
    scaling.SetScalingVariables(curr.x);

    // Initialise the solver with the new iterates x and z
    InitialiseAuxiliary(curr.x, curr.y, curr.z);
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
        UpdateTrustRegionNextState(trial);

        // Decrease the current value of the trial delta
        trial *= params.trust_region.delta_decrease;

        // Check if delta is now less than the allowed minimum
        if(trial < params.trust_region.delta_min)
            throw IPFilterErrorSearchDeltaNeighborhood();

        // Check if the current delta results results in any IEEE floating point exception
        if(AnyFloatingPointException(next))
            continue;

        // Prevent further trial delta sizes if the neighbourhood search algorithm is not active
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
        if(delta < params.trust_region.delta_min)
            throw IPFilterErrorSearchDeltaTrustRegion();

        // Calculate the linear model at the current and next states
        const double curr_m = curr.psi;
        const double next_m = CalculateNextLinearModel();

        // Calculate the ratio of decrease in actual and predicted optimality
        const double rho = (curr.psi - next.psi)/(curr_m - next_m);

        // Check if w(delta) pass the filter condition
        if(PassFilterCondition())
        {
            if(curr_m - next_m < params.trust_region.kappa*curr.theta*curr.theta)
            {
                // Extend the filter with the current (theta, psi) pair
                ExtendFilter();

                // Update the current maximum trust-region radius
                delta_initial = delta * params.trust_region.delta_increase;

                // Trial point has been found: leave loop
                break;
            }

            if(rho > params.trust_region.eta1)
            {
                // Update the current maximum trust-region radius
                delta_initial = (rho > params.trust_region.eta2) ?
                    delta * params.trust_region.delta_increase : delta;

                // Trial point has been found: leave loop
                break;
            }
        }

        // Decrease the trust-region radius
        UpdateTrustRegionNextState(params.trust_region.delta_decrease * delta);
    }

    // Update the neighborhood parameter M
    if(next.theta > next.mu * params.neighbourhood.epsilon * M)
        M = std::max(params.neighbourhood.Mmax, params.neighbourhood.alpha*next.theta/next.mu);

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
        if(rho > params.restoration.xi2) delta_initial = delta * params.trust_region.delta_increase;

        // Check if rho satisfies the Cauchy condition of the restoration algorithm
        if(rho > params.restoration.xi1)
            break;

        // Decrease the current trust-region radius and update all delta dependent quantities
        UpdateTrustRegionNextState(delta * params.trust_region.delta_decrease);

        // Check if delta is now less than the allowed minimum
        if(delta < params.trust_region.delta_min)
            throw IPFilterErrorSearchDeltaTrustRegionRestoration();
    }
}

void IPFilterSolver::SolveRestoration()
{
    // Output a message indicating the start of the restoration algorithm
    outputter.OutputMessage("...beginning the restoration algorithm");

    // Update the number of times the calculation entered the restoration phase algorithm
    ++result.num_restorations;

    // Extend the filter with the current (theta, psi) pair
    ExtendFilter();

    // Store the current value of the trust-region radius that was a result of the main iterations so far
    double delta_main = delta;

    while(true)
    {
        // Calculate the new normal and tangential steps for the restoration algorithm
        UpdateTrustRegionStepsRestoration();

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
}

void IPFilterSolver::UpdateNewtonNextState()
{
    // Update the step lengths of the x and z Newton steps
    alphax = CalculateLargestBoundaryStep(curr.x, dx);
    alphaz = CalculateLargestBoundaryStep(curr.z, dz);

    // Correct the previous calculated step lengths
    alphax = (alphax > 1.0) ? 1.0 : 0.995 * alphax;
    alphaz = (alphaz > 1.0) ? 1.0 : 0.995 * alphaz;

    // Update the next iterates
    next.x.noalias() = curr.x + alphax * dx;
    next.y.noalias() = curr.y + dy;
    next.z.noalias() = curr.z + alphaz * dz;

    // Update the next state
    UpdateState(next.x, next.y, next.z, next);
}

void IPFilterSolver::UpdateNewtonSteps()
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

    // Calculate the perturbation parameter
    mu = std::min(curr.mu, options.tolerance1*params.newton.factor);

    // Assemble the rhs vector of the linear system
    rhs.segment(0, n).noalias() = - Lx + mu*curr.x.cwiseInverse() - curr.z;
    rhs.segment(n, m).noalias() = - curr.h.func;

    // Calculate the step
    u = lu.solve(rhs);

    // Extract the x and y components of the normal step
    dx.noalias() = u.segment(0, n);
    dy.noalias() = u.segment(n, m);

    // Calculate the z components of the normal and tangential steps
    dz = -(curr.z.array() * dx.array() + curr.x.array() * curr.z.array() - mu)/curr.x.array();
}

void IPFilterSolver::UpdateState(const VectorXd& x, const VectorXd& y, const VectorXd& z, IPFilterState& state)
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

    // Update the maximum error among the feasibility, centrality and optimality errors
    state.error = std::max({state.errorh, state.errorc, state.errorl});

    // Update the residual error defined as the norm of the KKT equations
    state.residual = std::sqrt(state.thl*state.thl + state.thh*state.thh + x.cwiseProduct(z).squaredNorm());
}

void IPFilterSolver::UpdateTrustRegionNextState(double delta_)
{
    // Update the trust-region radius
    delta = delta_;

    // Update the normal and tangencial step lengths
    alphan = std::min(1.0, delta/norm_sn);
    alphat = std::min(alphan, delta/norm_st);

    // Update the iterates x(delta), y(delta), and z(delta)
    next.x.noalias() = curr.x + alphan * snx + alphat * stx;
    next.y.noalias() = curr.y + alphan * sny + alphat * sty;
    next.z.noalias() = curr.z + alphan * snz + alphat * stz;

    UpdateState(next.x, next.y, next.z, next);
}

void IPFilterSolver::UpdateTrustRegionSafeTangentialStep()
{
    // Define some auxiliary variables
    const unsigned n = dimx;
    const unsigned m = dimy;

    // Calculate the value of parameter sigma for the safe tangential step re-calculation
    const double sigma = CalculateSigmaSafeStep();

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

void IPFilterSolver::UpdateTrustRegionSteps()
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

void IPFilterSolver::UpdateTrustRegionStepsRestoration()
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
    const double sigma = CalculateSigmaRestoration();

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

} /* namespace Optima */
