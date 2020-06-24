// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "BasicSolver.hpp"

// C++ includes
#include <cassert>
#include <vector>

// Optima includes
#include <Optima/Analysis.hpp>
#include <Optima/Stepper.hpp>
#include <Optima/Exception.hpp>
#include <Optima/Options.hpp>
#include <Optima/Outputter.hpp>
#include <Optima/Result.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {
namespace {

/// The boolean flag that indicates a step failed.
const auto FAILED = false;

} // anonymous namespace

/// The implementation of the solver for basic optimization problems.
struct BasicSolver::Impl
{
    //======================================================================
    // DATA INITIALIZED AT CONSTRUCTION TIME
    //======================================================================

    Stepper stepper; ///< The calculator of the Newton step (dx, dy).
    Matrix A;        ///< The constant coefficient matrix A in the linear equality constraints in Ax = b.
    Index n;         ///< The number of variables in x.
    Index mb;        ///< The number of linear equality constraints in Ax = b.
    Index mh;        ///< The number of non-linear equality constraints in h(x) = 0.
    Index m;         ///< The number of constraints in Ax = b and h(x) = 0.
    Index t;         ///< The total number of variables in (x, y).

    //======================================================================
    // DATA INITIALIZED AT THE BEGINNING OF EACH SOLVE OPERATION
    //======================================================================

    ObjectiveFunction objectivefn;   ///< The objective function *f(x)* of the basic optimization problem.
    ConstraintFunction constraintfn; ///< The nonlinear equality constraint function *h(x)*.
    Vector b;                        ///< The right-hand side vector *b* of the linear equality constraints *Ax = b*.
    Vector xlower;                   ///< The lower bounds of the primal variables.
    Vector xupper;                   ///< The upper bounds of the primal variables.
    Stability stability;             ///< The stability state of the primal variables *x*.

    //======================================================================
    // DATA THAT IS CHANGED IN EVERY ITERATION OF THE ALGORITHM
    //======================================================================

    Result result;     ///< The result of the optimization problem.
    Analysis analysis; ///< The data for convergence analysis of the optimization calculation.
    Vector h;          ///< The result of the non-linear equality constraint function h(x).
    Matrix J;          ///< The Jacobian matrix J of the non-linear equality constraint function h(x).
    double f;          ///< The evaluated objective function f(x).
    Vector g;          ///< The evaluated gradient of the objective function f(x).
    Matrix H;          ///< The evaluated Hessian of the objective function f(x).
    Vector dx;         ///< The Newton step for the primal variables *x*.
    Vector dy;         ///< The Newton step for the Lagrange multipliers *y*.
    Vector rx;         ///< The residuals of the first-order optimality conditions.
    Vector ry;         ///< The residuals of the linear/nonlinear feasibility conditions.
    Vector ex;         ///< The relative errors of the first-order optimality conditions.
    Vector ey;         ///< The relative errors of the linear/nonlinear feasibility conditions.
    Vector x;          ///< The current value of x.
    Vector y;          ///< The current value of y.
    Vector z;          ///< The current value of z = g + tr(W)y.
    Vector xtrial;     ///< The trial iterate x(trial).
    Vector ytrial;     ///< The trial iterate y(trial).
    Vector dxtrial;    ///< The trial Newton step dx(trial).
    Vector dytrial;    ///< The trial Newton step dy(trial).
    double L;          ///< The current value L(x, y) of the Lagrange function.
    double E;          ///< The current error E(x, y) = ||grad(L)||^2.

    //======================================================================
    // OTHER DATA
    //======================================================================

    Options options;     ///< The options for the optimization calculation.
    Outputter outputter; ///< The outputter object to output computation state.

    /// Construct a BasicSolver::Impl instance with given details of the optimization problem.
    Impl(BasicSolverInitArgs args)
    : stepper({ args.n, args.m, args.A }), A(args.A)
    {
        // Initialize the members related to number of variables and constraints
        n  = args.n;
        m  = args.m;
        mb = args.A.rows();
        mh = m - mb;
        t  = n + m;

        // Ensure consistent dimensions of vectors/matrices.
        assert(n > 0);
        assert(mb <= m);

        // Allocate memory
        h       = zeros(mh);
        J       = zeros(mh, n);
        g       = zeros(n);
        H       = zeros(n, n);
        dx      = zeros(n);
        dy      = zeros(m);
        rx      = zeros(n);
        ry      = zeros(m);
        ex      = zeros(n);
        ey      = zeros(m);
        x       = zeros(n);
        y       = zeros(m);
        z       = zeros(n);
        xtrial  = zeros(n);
        ytrial  = zeros(m);
        dxtrial = zeros(n);
        dytrial = zeros(m);
    }

    /// Set the options for the optimization calculation.
    auto setOptions(const Options& opts) -> void
    {
    	// Set member options
    	options = opts;

        // Set the options of the optimization stepper
        stepper.setOptions(options);

        // Set the options of the outputter
        outputter.setOptions(options.output);
    }

    /// Solve the optimization problem.
    auto solve(BasicSolverSolveArgs args) -> Result
    {
        // Start timing the calculation
        Timer timer;

        // Finish the calculation if the problem has no variable
        if(n == 0)
        {
            result.succeeded = true;
            result.time = timer.elapsed();
            return result;
        }

        const auto maxiters = options.max_iterations;

        // Auxiliary references to some result variables
        auto& iterations = result.iterations;
        auto& succeeded = result.succeeded = false;

        // Perform initialization step. If not successful, exit the iterative algorithm.
        if(initialize(args) == FAILED)
        {
            result.succeeded = false;
            result.failure_reason = "Given initial guess causes the objective "
                "function and/or constraint function to fail. "
                "You need to provide a different initial guess.";
            result.time = timer.elapsed();
            return result;
        }

        outputHeaderTop();

        for(iterations = 0; iterations <= maxiters && !succeeded; ++iterations)
        {
            updateResiduals(); // TODO: This needs to stop depending on H so that the previous evaluateObjectiveFn does not compute H in case it is not needed.
            outputCurrentState();

            if((succeeded = converged()))
                break;

            if(applyNewtonStep() == FAILED)
                if(applySteepestDescentStep() == FAILED)
                    break;
        }

        outputHeaderBottom();

        finalize(args);

        // Finish timing the calculation
        result.time = timer.elapsed();

        return result;
    }

    /// Compute the sensitivity derivatives of the optimal solution.
    auto sensitivities(BasicSolverSensitivitiesArgs args) -> Result
    {
        Timer timer;

        auto [dgdp, dhdp, dbdp, stability, dxdp, dydp, dzdp] = args;

        stepper.sensitivities({ dgdp, dhdp, dbdp, stability, dxdp, dydp, dzdp });

        result.time_sensitivities = timer.elapsed();

        return result;
    }

    // Output the header at the top of the output table.
    auto outputHeaderTop() -> void
    {
        if(!options.output.active) return;
        outputter.addEntry("Iteration");
        outputter.addEntry("f(x)");
        outputter.addEntry("L(x, y)");
        outputter.addEntry("E(x, y)");
        outputter.addEntry("Optimality");
        outputter.addEntry("Feasibility");
        outputter.addEntries(options.output.xprefix, n, options.output.xnames);
        outputter.addEntries(options.output.yprefix, m, options.output.ynames);
        outputter.addEntries(options.output.zprefix, n, options.output.xnames);
        outputter.addEntries("r", n, options.output.xnames);
        outputter.outputHeader();
    };

    // Output the header at the bottom of the output table.
    auto outputHeaderBottom() -> void
    {
        if(!options.output.active) return;
        outputter.outputHeader();
    };

    // Output the current state of the solution.
    auto outputCurrentState() -> void
    {
        if(!options.output.active) return;
        outputter.addValue(result.iterations);
        outputter.addValue(f);
        outputter.addValue(L);
        outputter.addValue(E);
        outputter.addValue(result.error_optimality);
        outputter.addValue(result.error_feasibility);
        outputter.addValues(x);
        outputter.addValues(y);
        outputter.addValues(z);
        outputter.addValues(abs(rx));
        outputter.outputState();
    };

    // Initialize internal state before calculation begins
    auto initialize(BasicSolverSolveArgs args) -> bool
	{
        // Ensure the objective function has been given.
        Assert(args.obj != nullptr,
            "Could not proceed with BasicSolver::solve.",
                "No objective function given.");

        // Ensure the objective function has been given if number of nonlinear constraints is positive.
        Assert(mh == 0 or args.h != nullptr,
            "Could not proceed with BasicSolver::solve.",
                "No constraint function given.");

        // Ensure consistent dimensions of vectors/matrices.
        assert(args.x.rows() == n);
        assert(args.y.rows() == m);
        assert(args.b.rows() == mb);
        assert(args.xlower.rows() == n);
        assert(args.xupper.rows() == n);

        // Clear previous state of the Outputter instance
        outputter.clear();

        // Initialize the problem related data-members
        objectivefn  = args.obj;
        constraintfn = args.h;
        b            = args.b;
        xlower       = args.xlower;
        xupper       = args.xupper;

        // Initialize vectors x and y with given initial guesses
        x = args.x;
        y = args.y;

        // Initialize the stability of the variables with given initial state
        stability = args.stability;

        // Ensure the initial guess for `x` does not violate lower/upper bounds
        x.noalias() = max(args.x, xlower);
        x.noalias() = min(args.x, xupper);

    	// Initialize the Newton step calculator once for the upcoming decompose/solve calls
        stepper.initialize({ b, xlower, xupper, x, stability });

        // Initialize the convergence analysis data of the optimization calculation
        analysis.initialize(options.max_iterations);

        // Evaluate the objective function f and its gradient g at x0 (initial guess)
        const auto fres = evaluateObjectiveFn(x, { .f=true, .g=true, .H=true }); // TODO: The Hessian computation at this point should be eliminated in the future. Check how this can be done safely.

        // Return false if objective function evaluation failed.
        if(fres.failed) return false;

        // Evaluate the equality constraint function h at x0 (initial guess)
        const auto hres = evaluateConstraintFn(x);

        // Return false if constraint function evaluation failed.
        if(hres.failed) return false;

        // Return true as initialize step was successful.
        return true;
	}

    // Finalize the calculation by setting back computed state.
    auto finalize(BasicSolverSolveArgs args) -> void
	{
        // Set back in args the solution state of variables x, y, z
        args.x = x;
        args.y = y;
        args.z = z;

        // Set back in args the obtained stability state of the variables
        args.stability = stability;
    }

    // Evaluate the objective function.
    auto evaluateObjectiveFn(const Vector& x, ObjectiveRequirement req) -> ObjectiveResult
	{
        // Start time measuring.
    	Timer timer;

        // Create an ObjectiveResult with f, g, H to be filled
        ObjectiveResult res(f, g, H);

        // The computation requirements for f, g, and H
        res.requires = req;

        // Evaluate the objective function f(x)
        objectivefn(x, res);

        // Check the objective function produces finite numbers at this point
        if(res.requires.f && !std::isfinite(f))
            res.failed = true;
        if(res.requires.g && !g.allFinite())
            res.failed = true;
        if(res.requires.H && !H.allFinite())
            res.failed = true;

        // Update the number of objective function calls
        result.num_objective_evals   += 1;
        result.num_objective_evals_f += res.requires.f;
        result.num_objective_evals_g += res.requires.g;
        result.num_objective_evals_H += res.requires.H;

        // Update the time spent on objective function evaluation
        result.time_objective_evals += timer.elapsed();

        return res;
	}

    // Evaluate the equality constraint function h(x)
    auto evaluateConstraintFn(const Vector& x) -> ConstraintResult
	{
        // Start time measuring.
    	Timer timer;

        // Create a ConstraintResult with h and J to be filled
        ConstraintResult res{h, J};

        // Skip if there are no non-linear equality constraints
        if(mh == 0)
            return res;

        // Evaluate the constraint function h(x)
        constraintfn(x, res);

        // Check the constraint function produces finite numbers at this point
        if(!h.allFinite() || !J.allFinite())
            res.failed = true;

        // Update the time spent on constraint function evaluation
        result.time_constraint_evals += timer.elapsed();

        return res;
	}

	// Update the optimality, feasibility and complementarity errors
	auto updateResiduals() -> void
	{
    	// Canonicalize the W = [A; J] matrix as a pre-step to calculate the Newton step
        stepper.canonicalize({ x, y, g, H, J, xlower, xupper, stability });

        // Compute the current optimality and feasibility residuals
        stepper.residuals({ x, y, b, h, g, rx, ry, ex, ey, z });

		// Update the current optimality, feasibility and complementarity errors
		result.error_optimality  = norminf(ex);
		result.error_feasibility = norminf(ey);

		// Update the current maximum error of the optimization calculation
		result.error = std::max({
			result.error_optimality,
			result.error_feasibility
		});

        // Update the Lagrange function L(x, y) = f + tr(yb)*(Ax - b) + tr(yh)*h
        const auto yb = y.head(mb);
        const auto yh = y.tail(mh);

        L = f + yb.dot(A*x - b) + yh.dot(h);

        // Update the error E(x, y) = ||g + tr(W)y||^2 + ||Ax - b||^2 + ||h(x)||^2.
        // E = rx.squaredNorm() + ry.squaredNorm();
        // E = ex.squaredNorm() + ey.squaredNorm();
        E = computeError(x, y);

        // Store both L and E in the analysis container
        analysis.L.push_back(L);
        analysis.E.push_back(E);
	}

	// Compute the Lagrange function L(x, y) used in the line search algorithm.
	auto computeLagrangeFn(const Vector& x, const Vector& y) -> double
    {
        // Evaluate the objective function f(x) and its gradient g(x)
        const auto fres = evaluateObjectiveFn(x, { .f=true, .g=true, .H=true });

        // Return +inf if objective function evaluation failed.
        if(fres.failed) return infinity();

        // Evaluate the equality constraint function h(x)
        const auto hres = evaluateConstraintFn(x);

        // Return +inf if constraint function evaluation failed.
        if(hres.failed) return infinity();

        // Update the Lagrange function L(x, y) = f + tr(yb)*(Ax - b) + tr(yh)*h
        const auto yb = y.head(mb);
        const auto yh = y.tail(mh);

        L = f + yb.dot(A*x - b) + yh.dot(h);

        const auto L0 = analysis.L.front();

        return L - L0;
    }

	// Compute the error function E(x, y) used in the line search algorithm.
	auto computeError(const Vector& x, const Vector& y) -> double
    {
        // Evaluate the objective function f(x) and its gradient g(x)
        const auto fres = evaluateObjectiveFn(x, { .f=true, .g=true, .H=false });

        // Return +inf if objective function evaluation failed.
        if(fres.failed) return infinity();

        // Evaluate the equality constraint function h(x)
        const auto hres = evaluateConstraintFn(x);

        // Return +inf if constraint function evaluation failed.
        if(hres.failed) return infinity();

        // Compute the current optimality and feasibility residuals.
        // This can be achieved with Stepper::steepestDescent method.
        stepper.steepestDescent({ x, y, b, h, g, rx, ry });

		// Return the error E(x, y) = ||g + tr(W)y||^2 + ||Ax - b||^2 + ||h(x)||^2.
		return rx.squaredNorm() + ry.squaredNorm();
    }

    // The function that computes the Newton step
    auto computeNewtonStep() -> bool
    {
        // Start time measuring.
    	Timer timer;

        // Evaluate the Hessian of the objective function
        const auto fres = evaluateObjectiveFn(x, { .f=true, .g=true, .H=true });

        // Ensure the Hessian computation was successul.
        if(fres.failed)
            return false;

        // Canonicalize the W = [A; J] matrix as a pre-step to calculate the Newton step
        stepper.canonicalize({ x, y, g, H, J, xlower, xupper, stability });

    	// Decompose the Jacobian matrix and calculate a Newton step
        stepper.decompose({ x, y, g, H, J, xlower, xupper, stability });

        // Calculate the Newton step
        stepper.solve({ x, y, b, h, g, H, stability, dx, dy });

        // Update the time spent in linear systems
		result.time_linear_systems += timer.elapsed();

        // Newton step was calculated successfully.
        return true;
    };

	// Update the variables (x, y) with a Newton step.
	auto applyNewtonStep() -> bool
    {
        // Compute the Newton steps dx and dy
        if(computeNewtonStep() == FAILED)
            return FAILED;

        // Compute x(trial) and y(trial) taking care of the bounds of x
        xtrial = x;
        performAggressiveStep(xtrial, dx, xlower, xupper);
        ytrial = y + dy;

        // Compute the new error E after Newton step approach
        const auto Enew = computeError(xtrial, ytrial);

        // Get the error E at the initial guess
        const auto E0 = analysis.E.front();

        // Return true if new error is less than error at initial guess
        // if(Enew > E0)
        //     return FAILED;
        // if(Enew > E0)
        if(Enew > E)
        {
            xtrial = x;
            const auto alphamax = performConservativeStep(xtrial, dx, xlower, xupper);

            Vector xbar = xtrial;
            Vector ybar = y + alphamax * dy;

            // The phi(alpha) function that we want to minimize.
            const auto phi = [&](double alpha) -> double
            {
                // Vector xnext = x*(1 - alpha) + alpha*xbar;
                // Vector ynext = y*(1 - alpha) + alpha*ybar;
                // return computeError(xnext, ynext);

                xtrial.noalias() = x*(1 - alpha) + alpha*xbar;
                ytrial.noalias() = y*(1 - alpha) + alpha*ybar;
                return computeError(xtrial, ytrial);

                // xtrial.noalias() = x*(1 - alpha) + alpha*xbar;
                // ytrial.noalias() = y*(1 - alpha) + alpha*ybar;
                // Vector xnext = x + alpha*dx;
                // Vector ynext = y + alpha*dy;
            };

            std::cout << "alpha    = ";
            for(auto i = 0; i <= 100; ++i)
                std::cout << std::left << std::setw(10) << i/100.0 << " ";
            std::cout << std::endl;
            std::cout << "E(alpha) = ";
            for(auto i = 0; i <= 100; ++i)
                std::cout << std::left << std::setw(10) << phi(i/100.0) << " ";
            std::cout << std::endl;

            // Minimize the error phi(alpha) along the current Newton
            // direction. This is to be performed in the interval [0, 1], where
            // alpha=1 conincides with the largest Newton step that can be made
            // so that no lower/upper bound is violated.
            const auto lstol = options.linesearch.tolerance;
            const auto lsmaxiters = options.linesearch.maxiters;
            const auto alphamin = minimizeBrent(phi, 0.0, 1.0, lstol, lsmaxiters);

            // Calculate x(trial) and y(trial) using the minimizer alpha value
            xtrial.noalias() = x*(1 - alphamin) + alphamin*xbar;
            ytrial.noalias() = y*(1 - alphamin) + alphamin*ybar;

            // Compute the new error E after the steepest descent minmization approach
            const auto Enew2 = computeError(xtrial, ytrial);

            std::cout << "alpha*    = " << alphamin << std::endl;
            std::cout << "E(alpha*) = " << Enew2 << std::endl;

            auto i = 0;

            // return FAILED;
        }

        // Update x and y to their respective trial states
        x = xtrial;
        y = ytrial;

        // The Newton step approach was successful.
        return true;
    }

	// Update the variables (x, y) with a steepest descent step.
	auto applySteepestDescentStep() -> bool
    {
        // Compute the steepest descent steps dx and dy
        stepper.steepestDescent({ x, y, b, h, g, dx, dy });

        xtrial = x;
        const auto alphamax = performConservativeStep(xtrial, dx, xlower, xupper);

        Vector xbar = xtrial;
        Vector ybar = y + alphamax * dy;

        // The phi(alpha) function that we want to minimize.
        const auto phi = [&](double alpha) -> double
        {
            // xtrial.noalias() = x*(1 - alpha) + alpha*xbar;
            // ytrial.noalias() = y*(1 - alpha) + alpha*ybar;
            Vector xnext = x*(1 - alpha) + alpha*xbar;
            Vector ynext = y*(1 - alpha) + alpha*ybar;
            // Vector xnext = x + alpha*dx;
            // Vector ynext = y + alpha*dy;
            // return computeError(xtrial, ytrial);
            return computeLagrangeFn(xnext, ynext);
        };

        // std::cout << "E(alpha) = ";
        // for(auto i = 0; i <= 100; ++i)
        //     std::cout << phi(i/100.0) << " ";
        // std::cout << std::endl;

        std::cout << "L(p + alpha*dp) - L0 = ";
        for(auto i = 0; i <= 100; ++i)
            std::cout << phi(i/100.0) << " ";
        std::cout << std::endl;

        // Minimize the error E along the current steepest descent direction.
        auto alpha = minimizeGoldenSectionSearch(phi, 0.0, alphamax, 1e-40);

        // Calculate x(trial) and y(trial) using the minimizer alpha value
        xtrial.noalias() = x*(1 - alpha) + alpha*xbar;
        ytrial.noalias() = y*(1 - alpha) + alpha*ybar;

        // Compute the new error E after the steepest descent minmization approach
        // const auto Enew = computeError(xtrial, ytrial);
        const auto Lnew = computeLagrangeFn(xtrial, ytrial);

        // Get the error E at the initial guess
        // const auto E0 = analysis.E.front();

        // Return true if new error is less than error at initial guess
        // if(Enew > E0)
        //     return false;
        if(Lnew > 0.0)
            return false;

        // Update x and y to their respective trial states
        x = xtrial;
        y = ytrial;

        // The steepest descent approach was successful.
        return true;
    }

	/// Return true if the calculation converged.
    auto converged() const -> bool
    {
        // Prevent successfull convergence if linear equality constraints have not converged yet
        if(result.error_feasibility > options.tolerance_linear_equality_constraints)
            return false;

        // Check if the calculation should stop based on optimality/feasibility errors
        return result.error < options.tolerance;
    };
};

BasicSolver::BasicSolver(BasicSolverInitArgs args)
: pimpl(new Impl(args))
{}

BasicSolver::BasicSolver(const BasicSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

BasicSolver::~BasicSolver()
{}

auto BasicSolver::operator=(BasicSolver other) -> BasicSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto BasicSolver::setOptions(const Options& options) -> void
{
	pimpl->setOptions(options);
}

auto BasicSolver::solve(BasicSolverSolveArgs args) -> Result
{
    return pimpl->solve(args);
}

auto BasicSolver::sensitivities(BasicSolverSensitivitiesArgs args) -> Result
{
    return pimpl->sensitivities(args);
}

} // namespace Optima
