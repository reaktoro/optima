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
namespace detail {

// auto applyStep(VectorConstRef p, VectorConstRef dp, VectorConstRef plower, VectorConstRef pupper, double tau)
// {
//     auto ilimiting = p.size();
//     auto alpha = 1.0;
//     const auto np = p.size();
//     for(auto i = 0; i < np; ++i)
//     {
//         const auto qi = p[i] + alpha*dp[i];
//         if(qi <= plower[i])
//             alpha = (plower[i] - p[i])/dp[i]; // p[i] + alpha*dp[i] === plower[i]
//         if(qi <= plower[i] || qi >= pupper[i])
//     }

// }

} // namespace detail

/// The implementation of the solver for basic optimization problems.
struct BasicSolver::Impl
{
    /// The calculator of the Newton step (dx, dy, dz, dw)
    Stepper stepper;

    /// The options for the optimization calculation
    Options options;

    /// The result of the non-linear equality constraint function h(x)
    Vector h;

    /// The Jacobian matrix J of the non-linear equality constraint function h(x)
    Matrix J;

    /// The evaluated objective function f(x).
    double f;

    /// The evaluated gradient of the objective function f(x).
    Vector g;

    /// The evaluated Hessian of the objective function f(x).
    Matrix H;

    /// The result of the optimization problem
    Result result;

    /// The Newton step for the primal variables *x*.
    Vector dx;

    /// The Newton step for the Lagrange multipliers *y*.
    Vector dy;

    /// The residuals of the first-order optimality conditions.
    Vector rx;

    /// The residuals of the linear/nonlinear feasibility conditions.
    Vector ry;

    /// The trial iterate x(trial)
    Vector xtrial;

    /// The trial Newton step dx(trial)
    Vector dxtrial;

    /// The number of variables
    Index n;

    /// The number of linear equality constraints in Ax = b.
    Index mb;

    /// The number of non-linear equality constraints in h(x) = 0.
    Index mh;

    /// The number of constraints.
    Index m;

    /// The total number of variables (x, y, z, w).
    Index t;

    /// The outputter instance
    Outputter outputter;

    /// Construct a default BasicSolver::Impl instance.
    Impl()
    {}

    /// Construct a BasicSolver::Impl instance with given details of the optimization problem.
    Impl(BasicSolverInitArgs args)
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
        xtrial  = zeros(n);
        dxtrial = zeros(n);

        // Initialize step calculator
        stepper = Stepper({n, m, args.A});
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

    // Output the header and initial state of the solution
    auto outputInitialState(BasicSolverSolveArgs args) -> void
    {
        if(!options.output.active) return;

        // Aliases to canonical variables
        const auto& x = args.x;
        const auto& y = args.y;
        const auto& z = args.z;

        outputter.addEntry("Iteration");
        outputter.addEntry("f(x)");
        outputter.addEntry("Error");
        outputter.addEntries(options.output.xprefix, n, options.output.xnames);
        outputter.addEntries(options.output.yprefix, m, options.output.ynames);
        outputter.addEntries(options.output.zprefix, n, options.output.xnames);
        outputter.addEntries("r", n, options.output.xnames);
        outputter.addEntry("Optimality");
        outputter.addEntry("Feasibility");

        outputter.outputHeader();
        outputter.addValue(result.iterations);
        outputter.addValue(f);
        outputter.addValue(result.error);
        outputter.addValues(x);
        outputter.addValues(y);
        outputter.addValues(z);
        outputter.addValues(abs(rx));
        outputter.addValue(result.error_optimality);
        outputter.addValue(result.error_feasibility);
        outputter.outputState();
    };

    // The function that outputs the current state of the solution
    auto outputCurrentState(BasicSolverSolveArgs args) -> void
    {
        if(!options.output.active) return;

        // Aliases to canonical variables
        const auto& x = args.x;
        const auto& y = args.y;
        const auto& z = args.z;

        outputter.addValue(result.iterations);
        outputter.addValue(f);
        outputter.addValue(result.error);
        outputter.addValues(x);
        outputter.addValues(y);
        outputter.addValues(z);
        outputter.addValues(abs(rx));
        outputter.addValue(result.error_optimality);
        outputter.addValue(result.error_feasibility);
        outputter.outputState();
    };

    // Initialize internal state before calculation begins
    auto initialize(BasicSolverSolveArgs args) -> void
	{
        // Auxiliary references
        auto x         = args.x;
        auto y         = args.y;
        auto xlower    = args.xlower;
        auto xupper    = args.xupper;
        auto iordering = args.iordering;

        // Ensure consistent dimensions of vectors/matrices.
        assert(x.rows() == n);
        assert(y.rows() == m);
        assert(xlower.rows() == n);
        assert(xupper.rows() == n);
        assert(iordering.rows() == n);

        // Clear previous state of the Outputter instance
        outputter.clear();

        // Ensure the initial guess for `x` does not violate lower/upper bounds
        x.noalias() = max(x, xlower);
        x.noalias() = min(x, xupper);

    	// Initialize the Newton step calculator once for the upcoming decompose/solve calls
        stepper.initialize({xlower, xupper});

        // Evaluate the objective function
        evaluateObjectiveFunction(args);

        // Evaluate the constraint function
        evaluateConstraintFunction(args);

        // Assert the objective function produces finite numbers at this point
        Assert(std::isfinite(f) && g.allFinite() && H.allFinite(),
			"Failure evaluating the objective function.", "The evaluation of "
			"the objective function at the entry point of the optimization "
			"calculation produced non-finite numbers, "
			"such as `nan` and/or `inf`.");

        // Compute the Newton step for the current state
        computeNewtonStep(args);

        // Update the optimality, feasibility and complementarity errors
        updateResultErrors();
	}

    // Evaluate the objective function
    auto evaluateObjectiveFunction(BasicSolverSolveArgs args) -> void
	{
        // Ensure objective function is not empty.
        assert(args.obj != nullptr);

        // Create an ObjectiveResult with f, g, H to be filled
        ObjectiveResult res(f, g, H);

        // Establish the current needs for the objective function evaluation
        res.requires.f = true;
        res.requires.g = true;
        res.requires.H = true;

        // Evaluate the objective function f(x)
        args.obj(args.x, res);
	}

    // Evaluate the equality constraint function
    auto evaluateConstraintFunction(BasicSolverSolveArgs args) -> void
	{
        // Skip if there are no non-linear equality constraints
        if(mh == 0)
            return;

        // Ensure constraint function is not empty.
        assert(args.h != nullptr);

        // Create a ConstraintResult with h and J to be filled
        ConstraintResult res{h, J};

        // Evaluate the constraint function h(x)
        args.h(args.x, res);
	}

    // The function that computes the Newton step
    auto computeNewtonStep(BasicSolverSolveArgs args) -> void
    {
    	Timer timer;

        // Auxiliary variables
        auto x         = args.x;
        auto y         = args.y;
        auto z         = args.z;
        auto b         = args.b;
        auto xlower    = args.xlower;
        auto xupper    = args.xupper;
        auto iordering = args.iordering;
        auto& nul      = args.nul;
        auto& nuu      = args.nuu;

        // Ensure consistent dimensions of vectors/matrices.
        assert(x.rows() == n);
        assert(y.rows() == m);
        assert(z.rows() == n);
        assert(b.rows() == mb);
        assert(xlower.rows() == n);
        assert(xupper.rows() == n);
        assert(iordering.rows() == n);

    	// Decompose the Jacobian matrix and calculate a Newton step
        stepper.decompose({ x, y, g, H, J, xlower, xupper, iordering, nul, nuu });

        // Calculate the Newton step
        stepper.solve({ x, y, b, h, g, H, dx, dy, rx, ry, z });

        // Update the time spent in linear systems
		result.time_linear_systems += timer.elapsed();
    };

	// Update the optimality, feasibility and complementarity errors
	auto updateResultErrors() -> void
	{
		// Update the current optimality, feasibility and complementarity errors
		result.error_optimality  = norminf(rx)/norminf(g);
		result.error_feasibility = norminf(ry);

		// Update the current maximum error of the optimization calculation
		result.error = std::max({
			result.error_optimality,
			result.error_feasibility
		});
	}

    // Update the variables (x, y, z, w) with a Newton stepping scheme
    auto applyNewtonStepping(BasicSolverSolveArgs args) -> void
    {
        switch(options.step) {
        case Aggressive: return applyNewtonSteppingAggressive(args);
        default: return applyNewtonSteppingConservative(args);
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

	// Update the variables (x, y, z, w) with an aggressive Newton stepping scheme
	auto applyNewtonSteppingAggressive(BasicSolverSolveArgs args) -> void
	{
        // Auxiliary variables
        auto x = args.x;
        auto y = args.y;
        auto xlower = args.xlower;
        auto xupper = args.xupper;

        // Ensure consistent dimensions of vectors/matrices.
        assert(x.rows() == n);
        assert(y.rows() == m);
        assert(xlower.rows() == n);
        assert(xupper.rows() == n);

		// Update xtrial with the calculated Newton step
		xtrial = x + dx;

        // Ensure no entry in `x` violate lower/upper bounds
        xtrial.noalias() = max(xtrial, xlower);
        xtrial.noalias() = min(xtrial, xupper);

		// Calculate the trial Newton step for the aggressive mode
		dxtrial = xtrial - x;

		// Update the x variables
		x = xtrial;

		// Update the y-Lagrange multipliers
		y += dy;
	};

	// Update the variables (x, y, z, w) with a conservative Newton stepping scheme
	auto applyNewtonSteppingConservative(BasicSolverSolveArgs args) -> void
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
        // Check if the calculation should stop based on optimality/feasibility errors
        return result.error < options.tolerance;
    };

    /// Solve the optimization problem.
    auto solve(BasicSolverSolveArgs args) -> Result
    {
        // Start timing the calculation
        Timer timer;

        // Ensure the objective function has been given.
        Assert(args.obj != nullptr,
            "Could not initialize BasicSolver.",
                "No objective function given.");

        // Ensure the objective function has been given if number of nonlinear constraints is positive.
        Assert(mh == 0 or args.h != nullptr,
            "Could not initialize BasicSolver.",
                "No constraint function given.");

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
        outputInitialState(args);

        for(iterations = 1; iterations <= maxiters && !succeeded; ++iterations)
        {
            applyNewtonStepping(args);
            outputCurrentState(args);

            if((succeeded = converged()))
                break;

            evaluateObjectiveFunction(args);
			computeNewtonStep(args);
			updateResultErrors();
        }

        // Output a final header
        outputter.outputHeader();

        // Finish timing the calculation
        result.time = timer.elapsed();

        return result;
    }

    /// Compute the sensitivity derivatives of the optimal solution.
    auto sensitivities(BasicSolverSensitivitiesArgs args) -> Result
    {
        Timer timer;

        auto [dgdp, dhdp, dbdp, dxdp, dydp, dzdp] = args;

        stepper.sensitivities({ dgdp, dhdp, dbdp, dxdp, dydp, dzdp} );

        result.time_sensitivities = timer.elapsed();

        return result;
    }
};

BasicSolver::BasicSolver()
: pimpl(new Impl())
{}

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
