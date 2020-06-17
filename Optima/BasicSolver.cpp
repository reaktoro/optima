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

// Optima includes
#include <Optima/Stepper.hpp>
#include <Optima/Exception.hpp>
#include <Optima/Options.hpp>
#include <Optima/Outputter.hpp>
#include <Optima/Result.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

/// The implementation of the solver for basic optimization problems.
struct BasicSolver::Impl
{
    //======================================================================
    // DATA INITIALIZED AT CONSTRUCTION TIME
    //======================================================================

    Stepper stepper; ///< The calculator of the Newton step (dx, dy).
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

    Result result;  ///< The result of the optimization problem
    Vector h;       ///< The result of the non-linear equality constraint function h(x)
    Matrix J;       ///< The Jacobian matrix J of the non-linear equality constraint function h(x)
    double f;       ///< The evaluated objective function f(x).
    Vector g;       ///< The evaluated gradient of the objective function f(x).
    Matrix H;       ///< The evaluated Hessian of the objective function f(x).
    Vector dx;      ///< The Newton step for the primal variables *x*.
    Vector dy;      ///< The Newton step for the Lagrange multipliers *y*.
    Vector rx;      ///< The residuals of the first-order optimality conditions.
    Vector ry;      ///< The residuals of the linear/nonlinear feasibility conditions.
    Vector x;       ///< The current value of x.
    Vector y;       ///< The current value of y.
    Vector z;       ///< The current value of z = g + tr(W)y.
    Vector xtrial;  ///< The trial iterate x(trial).
    Vector ytrial;  ///< The trial iterate y(trial).
    Vector dxtrial; ///< The trial Newton step dx(trial).
    Vector dytrial; ///< The trial Newton step dy(trial).

    //======================================================================
    // OTHER DATA
    //======================================================================

    Options options;     ///< The options for the optimization calculation.
    Outputter outputter; ///< The outputter object to output computation state.

    /// Construct a BasicSolver::Impl instance with given details of the optimization problem.
    Impl(BasicSolverInitArgs args)
    : stepper({ args.n, args.m, args.A })
    {
        // Initialize the members related to number of variables and constraints
        n  = args.n;
        m  = args.m;
        mb = args.A.rows();
        mh = m - mb;

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

        initialize(args);

        outputHeaderTop();

        for(iterations = 1; iterations <= maxiters && !succeeded; ++iterations)
        {
            evaluateObjectiveFn(x, { .f=true, .g=true, .H=true });
            evaluateConstraintFn(x);
			updateResiduals();
            outputCurrentState();

            if((succeeded = converged()))
                break;

			computeNewtonStep();
            applyNewtonStepping();
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
        outputter.addEntry("Optimality");
        outputter.addEntry("Feasibility");
        outputter.addEntry("Error");
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
        outputter.addValue(result.error_optimality);
        outputter.addValue(result.error_feasibility);
        outputter.addValue(result.error);
        outputter.addValues(x);
        outputter.addValues(y);
        outputter.addValues(z);
        outputter.addValues(abs(rx));
        outputter.outputState();
    };

    // Initialize internal state before calculation begins
    auto initialize(BasicSolverSolveArgs args) -> void
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

        return res;
	}

    // Evaluate the equality constraint function h(x)
    auto evaluateConstraintFn(const Vector& x) -> ConstraintResult
	{
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

        return res;
	}

	// Update the optimality, feasibility and complementarity errors
	auto updateResiduals() -> void
	{
    	// Canonicalize the W = [A; J] matrix as a pre-step to calculate the Newton step
        stepper.canonicalize({ x, y, g, H, J, xlower, xupper, stability });

        // Compute the current optimality and feasibility residuals
        stepper.residuals({ x, y, b, h, g, rx, ry, z });

		// Update the current optimality, feasibility and complementarity errors
		result.error_optimality  = norminf(rx)/(1 + norminf(g));
		result.error_feasibility = norminf(ry);

		// Update the current maximum error of the optimization calculation
		result.error = std::max({
			result.error_optimality,
			result.error_feasibility
		});
	}

    // The function that computes the Newton step
    auto computeNewtonStep() -> void
    {
    	Timer timer;

    	// Decompose the Jacobian matrix and calculate a Newton step
        stepper.decompose({ x, y, g, H, J, xlower, xupper, stability });

        // Calculate the Newton step
        stepper.solve({ x, y, b, h, g, H, stability, dx, dy });

        // Update the time spent in linear systems
		result.time_linear_systems += timer.elapsed();
    };

    // Update the variables (x, y, z, w) with a Newton stepping scheme
    auto applyNewtonStepping() -> void
    {
        switch(options.step) {
        case Aggressive: return applyNewtonSteppingAggressive();
        default: return applyNewtonSteppingConservative();
        }
    };

    /// Perform a backtracking line search algorithm to check if a shorter
    /// Newton step results in better residuals.
    auto applyBacktrackingLineSeach() -> void
	{
    	// Skip if first iteration

    	// TODO Use dxtrial to calculate alpha step length to decrease the stepping

//		// Establish the current needs for the objective function evaluation at the trial iterate
//		f.requires.f = true;
//		f.requires.g = false;
//		f.requires.H = false;
//
//		// Evaluate the objective function at the trial iterate


//		f = objective(xtrial);
//
//		args.f = f.f;
//		args.g = f.g;
//		args.H = f.H;
//
//		// Initialize the step length factor
//		double alpha = fractionToTheBoundary(x, dx, options.tau);
//
//		// The number of tentatives to find a trial iterate that results in finite objective result
//		unsigned tentatives = 0;
//
//		// Repeat until f(xtrial) is finite
//		while(!isfinite(f) && ++tentatives < 10)
//		{
//			// Calculate a new trial iterate using a smaller step length
//			xtrial = x + alpha * dx;
//
//			// Evaluate the objective function at the new trial iterate
//			f.requires.f = true;
//			f.requires.g = false;
//			f.requires.H = false;


//			f = objective(xtrial);
//
//			args.f = f.f;
//			args.g = f.g;
//			args.H = f.H;
//
//			// Decrease the current step length
//			alpha *= 0.5;
//		}
//
//		// Return false if xtrial could not be found s.t. f(xtrial) is finite
//		if(tentatives == 10)
//			return false;
//
//		// Update the iterate x from xtrial
//		x = xtrial;
//
//		// Update the gradient and Hessian at x
//		f.requires.f = false;
//		f.requires.g = true;
//		f.requires.H = true;


//		f = objective(x);
//
//		args.f = f.f;
//		args.g = f.g;
//		args.H = f.H;

	}

	// Compute the error function E(x, y) used in the line search algorithm.
	auto computeError(const Vector& x, const Vector& y) -> double
    {
        // Evaluate the objective function f(x) and its gradient g(x)
        evaluateObjectiveFn(x, {.f=true, .g=true, .H=false});

        // Evaluate the equality constraint function h(x)
        evaluateConstraintFn(x);

        // Compute the current optimality and feasibility residuals
        stepper.residuals({ x, y, b, h, g, rx, ry, z });

		// Update the current optimality, feasibility and complementarity errors
		const auto error_optimality  = norminf(rx)/(1 + norminf(g));
		const auto error_feasibility = norminf(ry);

		// Return the maximum error
		return std::max({ error_optimality, error_feasibility });
    }

	// Update the variables (x, y, z, w) with an aggressive Newton stepping scheme
	auto applyNewtonSteppingAggressive() -> void
	{
		// // Update xtrial with the calculated Newton step
		// xtrial = x + dx;

        // // Ensure no entry in `x` violate lower/upper bounds
        // xtrial.noalias() = max(xtrial, xlower);
        // xtrial.noalias() = min(xtrial, xupper);

        xtrial = x;

        const auto E0 = result.error;

        // performAggressiveStep(xtrial, dx, xlower, xupper);
        performConservativeStep(xtrial, dx, xlower, xupper);

        // computeError({ })



        // // Create an ObjectiveResult with f, g, H to be filled
        // ObjectiveResult res(f, g, H);

        // // Establish the current needs for the objective function evaluation
        // res.requires.f = true;
        // res.requires.g = true;
        // res.requires.H = false; // there is no need for Hessian updates during the line search

        // // Evaluate the objective function f(x)
        // args.obj(xtrial, res);

        // bool line_search_needed = false;

        // // If x(trial) produces non-finite values, line search is needed
        // line_search_needed = !std::isfinite(f) || g.allFinite();

        // error_new < error





		// Calculate the trial Newton step for the aggressive mode
		dxtrial = xtrial - x;

		// Update the x variables
		x = xtrial;

		// Update the y-Lagrange multipliers
		y += dy;
	};

	// Update the variables (x, y, z, w) with a conservative Newton stepping scheme
	auto applyNewtonSteppingConservative() -> void
	{
		// Aliases to variables x, y, z, w
		// VectorRef x = args.x;
		// VectorRef y = args.y;
		// VectorRef z = args.z;

		// // The indices of variables with lower/upper bounds and fixed values
		// IndicesConstRef ilower = constraints.variablesWithLowerBounds();
		// IndicesConstRef iupper = constraints.variablesWithUpperBounds();
		// IndicesConstRef ifixed = constraints.variablesWithFixedValues();

		// // Initialize the step length factor
		// double alphax = xStepLength(x, dx, xlower, xupper, tau);
		// double alphaz = zStepLength(z, dz, tau);
		// double alphaw = wStepLength(w, dw, tau);
		// double alpha = alphax;

		// // The number of tentatives to find a trial iterate that results in finite objective result
		// unsigned tentatives = 0;

		// // Repeat until a suitable xtrial iterate if found such that f(xtrial) is finite
		// for(; tentatives < 10; ++tentatives)
		// {
		// 	// Calculate the current trial iterate for x
		// 	xtrial = x + alpha * dx;

		// 	// Evaluate the objective function at the trial iterate
		// 	f.requires.f = true;
		// 	f.requires.g = false;
		// 	f.requires.H = false;


		// 	f = objective(xtrial);

		// 	args.f = f.f;
		// 	args.g = f.g;
		// 	args.H = f.H;

		// 	// Leave the loop if f(xtrial) is finite
		// 	if(isfinite(f))
		// 		break;

		// 	// Decrease alpha in a hope that a shorter step results f(xtrial) finite
		// 	alpha *= 0.01;
		// }

		// // Return false if xtrial could not be found s.t. f(xtrial) is finite
		// if(tentatives == 10)
		// 	return false;

		// // Update the iterate x from xtrial
		// x = xtrial;

		// // Update the z-Lagrange multipliers
		// z += alphaz * dz;

		// // Update the w-Lagrange multipliers
		// w += alphaw * dw;

		// // Update the y-Lagrange multipliers
		// y += dy;

		// // Update the gradient and Hessian at x
		// f.requires.f = false;
		// f.requires.g = true;
		// f.requires.H = true;


		// f = objective(x);

		// args.f = f.f;
		// args.g = f.g;
		// args.H = f.H;

		// // Return true as found xtrial results in finite f(xtrial)
		// return true;
	};

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
