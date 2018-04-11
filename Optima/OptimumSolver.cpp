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

#include "OptimumSolver.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IpSaddlePointMatrix.hpp>
#include <Optima/Objective.hpp>
#include <Optima/OptimumOptions.hpp>
#include <Optima/OptimumParams.hpp>
#include <Optima/OptimumResult.hpp>
#include <Optima/OptimumState.hpp>
#include <Optima/OptimumStepper.hpp>
#include <Optima/OptimumStructure.hpp>
#include <Optima/Outputter.hpp>
#include <Optima/Result.hpp>
#include <Optima/SaddlePointMatrix.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>
using namespace Eigen;

namespace Optima {
namespace {

auto isfinite(const ObjectiveResult& res) -> bool
{
    return std::isfinite(res.f) && res.g.allFinite();
}

//auto xStepLength(VectorConstRef x, VectorConstRef dx, VectorConstRef l, VectorConstRef u, double tau) -> double
//{
//    double alpha = 1.0;
//    const Index size = x.size();
//    for(Index i = 0; i < size; ++i)
//        alpha = (dx[i] < 0.0) ? std::min(alpha, tau*(l[i] - x[i])/dx[i]) :
//                (dx[i] > 0.0) ? std::min(alpha, tau*(u[i] - x[i])/dx[i]) : alpha;
//    return alpha;
//}
//
//auto zStepLength(VectorConstRef z, VectorConstRef dz, double tau) -> double
//{
//    double alpha = 1.0;
//    const Index size = z.size();
//    for(Index i = 0; i < size; ++i)
//        if(dz[i] < 0.0) alpha = std::min(alpha, -tau*z[i]/dz[i]);
//    return alpha;
//}
//
//auto wStepLength(VectorConstRef w, VectorConstRef dw, double tau) -> double
//{
//    double alpha = 1.0;
//    const Index size = w.size();
//    for(Index i = 0; i < size; ++i)
//        if(dw[i] > 0.0) alpha = std::min(alpha, -tau*w[i]/dw[i]);
//    return alpha;
//}

} // namespace

struct OptimumSolver::Impl
{
    /// The structure of the optimization problem
    OptimumStructure structure;

    /// The calculator of the Newton step (dx, dy, dz, dw)
    OptimumStepper stepper;

    /// The evaluation of the objective function.
    ObjectiveResult obj;

    /// The result of the optimization problem
    OptimumResult result;

    /// The trial iterate x(trial)
    Vector xtrial;

    /// The trial Newton step dx(trial)
    Vector dxtrial;

    /// The lower bounds for each variable x (-inf with no lower bound)
    Vector xlower;

    /// The upper bounds for each variable x (+inf with no lower bound)
    Vector xupper;

    /// The outputter instance
    Outputter outputter;

    /// The options for the optimization calculation
    OptimumOptions options;

    /// The number of variables
    Index n;

    /// The number free and fixed variables.
    Index nx, nf;

    /// The number of equality constraints.
    Index m;

    /// The total number of variables (x, y, z, w).
    Index t;

    /// Initialize the optimization solver with the structure of the problem.
    Impl(const OptimumStructure& structure)
    : structure(structure), stepper(structure)
    {
        // Initialize the members related to number of variables and constraints
        n = structure.numVariables();
        m = structure.numEqualityConstraints();

        // Initialize the number of fixed and free variables in x
        nf = structure.variablesWithFixedValues().size();
        nx = n - nf;

        // Allocate memory
        xtrial.resize(n);

        // Initialize xlower and xupper with -inf and +inf
        xlower = constants(n, -infinity());
        xupper = constants(n,  infinity());
    }

    /// Set the options for the optimization calculation.
    auto setOptions(const OptimumOptions& _options) -> void
    {
    	// Set member options
    	options = _options;

        // Set the options of the optimization stepper
        stepper.setOptions(options);

        // Set the options of the outputter
        outputter.setOptions(options.output);
    }

    // Output the header and initial state of the solution
    auto outputInitialState(const OptimumState& state) -> void
    {
        if(!options.output.active) return;

        outputter.addEntry("Iteration");
        outputter.addEntry("f(x)");
        outputter.addEntry("Error");
        outputter.addEntries(options.output.xprefix, n, options.output.xnames);
        outputter.addEntries(options.output.yprefix, m, options.output.ynames);
        outputter.addEntries(options.output.zprefix, n, options.output.xnames);
        outputter.addEntries(options.output.wprefix, n, options.output.xnames);
        outputter.addEntries("r", n, options.output.xnames);
        outputter.addEntry("Optimality");
        outputter.addEntry("Feasibility");
        outputter.addEntry("CentralityLower");
        outputter.addEntry("CentralityUpper");

        outputter.outputHeader();
        outputter.addValue(result.iterations);
        outputter.addValue(obj.f);
        outputter.addValue(result.error);
        outputter.addValues(state.x);
        outputter.addValues(state.y);
        outputter.addValues(state.z);
        outputter.addValues(state.w);
        outputter.addValues(abs(stepper.residual().a));
        outputter.addValue(result.error_optimality);
        outputter.addValue(result.error_feasibility);
        outputter.addValue(result.error_complementarity_lower);
        outputter.addValue(result.error_complementarity_upper);
        outputter.outputState();
    };

    // The function that outputs the current state of the solution
    auto outputCurrentState(const OptimumState& state) -> void
    {
        if(!options.output.active) return;

        outputter.addValue(result.iterations);
        outputter.addValue(obj.f);
        outputter.addValue(result.error);
        outputter.addValues(state.x);
        outputter.addValues(state.y);
        outputter.addValues(state.z);
        outputter.addValues(state.w);
        outputter.addValues(abs(stepper.residual().a));
        outputter.addValue(result.error_optimality);
        outputter.addValue(result.error_feasibility);
        outputter.addValue(result.error_complementarity_lower);
        outputter.addValue(result.error_complementarity_upper);
        outputter.outputState();
    };

    // Initialize internal state before calculation begins
    auto initialize(const OptimumParams& params, OptimumState& state) -> void
	{
        // Clear previous state of the Outputter instance
        outputter.clear();

        // The indices of variables with lower/upper bounds and fixed values
        IndicesConstRef ilower = structure.variablesWithLowerBounds();
        IndicesConstRef iupper = structure.variablesWithUpperBounds();
        IndicesConstRef ifixed = structure.variablesWithFixedValues();

        // Initialize xlower and xupper with the given lower and upper bounds
        xlower(ilower) = params.xlower;
        xupper(iupper) = params.xupper;

        // Ensure the initial guesses for x, y, z, w have proper dimensions
        if(state.x.size() != n) state.x = zeros(n);
        if(state.y.size() != m) state.y = zeros(m);
        if(state.z.size() != n) state.z = zeros(n);
        if(state.w.size() != n) state.w = zeros(n);

        // Ensure the initial guess for `x` does not violate lower/upper bounds
        for(Index i : ilower) state.x[i] = std::max(state.x[i], xlower[i] + options.mu);
        for(Index i : iupper) state.x[i] = std::min(state.x[i], xupper[i] - options.mu);

        // Ensure z = mu/(x - xlower) for variables with lower bounds
        for(Index i : ilower)
        	state.z[i] = state.x[i] == xlower[i] ?
        		1.0 : options.mu / (state.x[i] - xlower[i]);

//        state.z(ilower) = options.mu / ( state.x(ilower) - xlower(ilower) );

        // Ensure w = mu/(xupper - x) for variables with upper bounds
        for(Index i : iupper)
        	state.w[i] = state.x[i] == xupper[i] ?
        		1.0 : options.mu / (xupper[i] - state.x[i]);
//        state.w(iupper) = options.mu / ( xupper(iupper) - state.x(iupper) );

        // Set the values of x, z, w corresponding to fixed variables
        state.x(ifixed) = params.xfixed;
        state.z(ifixed).fill(0.0);
        state.w(ifixed).fill(0.0);

        // Evaluate the objective function
        evaluateObjectiveFunction(params, state);

        // Assert the objective function produces finite numbers at this point
        Assert(isfinite(obj),
			"Failure evaluating the objective function.", "The evaluation of "
			"the objective function at the entry point of the optimization "
			"calculation produced non-finite numbers, "
			"such as `nan` and/or `inf`.");

        // Compute the Newton step for the current state
        computeNewtonStep(params, state);

        // Update the optimality, feasibility and complementarity errors
        updateResultErrors();
	}

    // Evaluate the objective function
    auto evaluateObjectiveFunction(const OptimumParams& params, OptimumState& state) -> void
	{
        // Establish the current needs for the objective function evaluation
        obj.requires.f = true;
        obj.requires.g = true;
        obj.requires.H = true;

        // Evaluate the objective function
        obj = params.objective(state.x);

        // Update the state TODO This needs to be improved. OptimumState perhaps should have a ObjectiveState object instead. Thus, state.objective = params.objective(state.x)
        state.f = obj.f;
        state.g = obj.g;
        state.H = obj.H;
	}

    // The function that computes the Newton step
    auto computeNewtonStep(const OptimumParams& params, OptimumState& state) -> void
    {
    	Timer timer;

    	// Decompose the Jacobian matrix and calculate a Newton step
        Result res1 = stepper.decompose(params, state);

        // Assert the decomposition of the Jacobian matrix succeeded
        Assert(res1.success(), "Could not compute a Newton step.",
			"The decomposition of the Jacobian matrix failed.");

        // Calculate the Newton step
        Result res2 = stepper.solve(params, state);

        // Assert the calculation of the Newton step succeeded
        Assert(res2.success(), "Could not compute a Newton step.",
			"The decomposition of the Jacobian matrix succeeded, but the "
			"step calculation failed.");

        // Update the time spent in linear systems
		result.time_linear_systems += timer.elapsed();
    };

	// Update the optimality, feasibility and complementarity errors
	auto updateResultErrors() -> void
	{
		// Update the current optimality, feasibility and complementarity errors
		result.error_optimality            = norminf(stepper.residual().a);
		result.error_feasibility           = norminf(stepper.residual().b);
		result.error_complementarity_lower = norminf(stepper.residual().c);
		result.error_complementarity_upper = norminf(stepper.residual().d);

		// Update the current maximum error of the optimization calculation
		result.error = std::max({
			result.error_optimality,
			result.error_feasibility,
			result.error_complementarity_lower,
			result.error_complementarity_upper
		});
	}

    // Update the variables (x, y, z, w) with a Newton stepping scheme
    auto applyNewtonStepping(const OptimumParams& params, OptimumState& state) -> void
    {
        switch(options.step) {
        case Aggressive: return applyNewtonSteppingAggressive(params, state);
        default: return applyNewtonSteppingConservative(params, state);
        }
    };

    /// Perform a backtracking line search algorithm to check if a shorter
    /// Newton step results in better residuals.
    auto applyBacktrackingLineSeach() -> void
	{
    	// Skip if first iteration

    	// TODO Use dxtrial to calculate alpha step length to decrease the stepping

//		// Establish the current needs for the objective function evaluation at the trial iterate
//		obj.requires.f = true;
//		obj.requires.g = false;
//		obj.requires.H = false;
//
//		// Evaluate the objective function at the trial iterate
//		obj = params.objective(xtrial);
//
//		state.f = obj.f;
//		state.g = obj.g;
//		state.H = obj.H;
//
//		// Initialize the step length factor
//		double alpha = fractionToTheBoundary(x, dx, options.tau);
//
//		// The number of tentatives to find a trial iterate that results in finite objective result
//		unsigned tentatives = 0;
//
//		// Repeat until f(xtrial) is finite
//		while(!isfinite(obj) && ++tentatives < 10)
//		{
//			// Calculate a new trial iterate using a smaller step length
//			xtrial = x + alpha * dx;
//
//			// Evaluate the objective function at the new trial iterate
//			obj.requires.f = true;
//			obj.requires.g = false;
//			obj.requires.H = false;
//			obj = params.objective(xtrial);
//
//			state.f = obj.f;
//			state.g = obj.g;
//			state.H = obj.H;
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
//		obj.requires.f = false;
//		obj.requires.g = true;
//		obj.requires.H = true;
//		obj = params.objective(x);
//
//		state.f = obj.f;
//		state.g = obj.g;
//		state.H = obj.H;

	}

	// Update the variables (x, y, z, w) with an aggressive Newton stepping scheme
	auto applyNewtonSteppingAggressive(const OptimumParams& params, OptimumState& state) -> void
	{
		// Aliases to variables x, y, z, w
		VectorRef x = state.x;
		VectorRef y = state.y;
		VectorRef z = state.z;
		VectorRef w = state.w;

		// Aliases to Newton steps calculated before
		VectorConstRef dx = stepper.step().x;
		VectorConstRef dy = stepper.step().y;
		VectorConstRef dz = stepper.step().z;
		VectorConstRef dw = stepper.step().w;

        // The indices of variables with lower/upper bounds and fixed values
        IndicesConstRef ilower = structure.variablesWithLowerBounds();
        IndicesConstRef iupper = structure.variablesWithUpperBounds();
        IndicesConstRef ifixed = structure.variablesWithFixedValues();

		// Update xtrial with the calculated Newton step
		xtrial = x + dx;

		// Update x for variables with violated lower bounds
		for(Index i : ilower)
			if(xtrial[i] <= xlower[i])
				xtrial[i] = x[i] - (x[i] - xlower[i]) * options.tau;

		// Update x for variables with violated upper bounds
		for(Index i : iupper)
			if(xtrial[i] >= xupper[i])
				xtrial[i] = x[i] + (xupper[i] - x[i]) * options.tau;

		// Calculate the trial Newton step for the aggressive mode
		dxtrial = xtrial - x;

		// Update the z-Lagrange multipliers for variables with lower bounds
		for(Index i : ilower)
			z[i] += (z[i] + dz[i] > 0.0) ?
				dz[i] : -options.tau * z[i];

		// Update the w-Lagrange multipliers for variables with upper bounds
		for(Index i : iupper)
			w[i] += (w[i] + dw[i] > 0.0) ?
				dw[i] : -options.tau * w[i];

		// Update the x variables
		x = xtrial;


		for(Index i : ilower)
			if(xtrial[i] < xlower[i])
				RuntimeError("Failed to check lower bounds for variable " + std::to_string(i), "");


		for(Index i : iupper)
			if(xtrial[i] > xupper[i])
				RuntimeError("Failed to check upper bounds for variable " + std::to_string(i), "");

//		Assert((xtrial.array() > xlower).all(), "Failed to check lower bounds.", "There are variables with values below bounds.");
//		Assert((xtrial.array() < xupper).all(), "Failed to check lower bounds.", "There are variables with values below bounds.");

		// Update the y-Lagrange multipliers
		y += dy;

		// Set the values of x, z, w corresponding to fixed variables
		x(ifixed) = params.xfixed;
		z(ifixed).fill(0.0);
		w(ifixed).fill(0.0);
	};

	// Update the variables (x, y, z, w) with a conservative Newton stepping scheme
	auto applyNewtonSteppingConservative(const OptimumParams& params, OptimumState& state) -> void
	{
//		// Aliases to variables x, y, z, w
//		VectorRef x = state.x;
//		VectorRef y = state.y;
//		VectorRef z = state.z;
//		VectorRef w = state.w;
//
//		// Aliases to Newton steps calculated before
//		VectorConstRef dx = stepper.step().x;
//		VectorConstRef dy = stepper.step().y;
//		VectorConstRef dz = stepper.step().z;
//		VectorConstRef dw = stepper.step().w;
//
//		// The indices of variables with lower/upper bounds and fixed values
//		IndicesConstRef ilower = structure.variablesWithLowerBounds();
//		IndicesConstRef iupper = structure.variablesWithUpperBounds();
//		IndicesConstRef ifixed = structure.variablesWithFixedValues();
//
//		// Initialize the step length factor
//		double alphax = xStepLength(x, dx, xlower, xupper, tau);
//		double alphaz = zStepLength(z, dz, tau);
//		double alphaw = wStepLength(w, dw, tau);
//		double alpha = alphax;
//
//		// The number of tentatives to find a trial iterate that results in finite objective result
//		unsigned tentatives = 0;
//
//		// Repeat until a suitable xtrial iterate if found such that f(xtrial) is finite
//		for(; tentatives < 10; ++tentatives)
//		{
//			// Calculate the current trial iterate for x
//			xtrial = x + alpha * dx;
//
//			// Evaluate the objective function at the trial iterate
//			obj.requires.f = true;
//			obj.requires.g = false;
//			obj.requires.H = false;
//			obj = params.objective(xtrial);
//
//			state.f = obj.f;
//			state.g = obj.g;
//			state.H = obj.H;
//
//			// Leave the loop if f(xtrial) is finite
//			if(isfinite(obj))
//				break;
//
//			// Decrease alpha in a hope that a shorter step results f(xtrial) finite
//			alpha *= 0.01;
//		}
//
//		// Return false if xtrial could not be found s.t. f(xtrial) is finite
//		if(tentatives == 10)
//			return false;
//
//		// Update the iterate x from xtrial
//		x = xtrial;
//
//		// Update the z-Lagrange multipliers
//		z += alphaz * dz;
//
//		// Update the w-Lagrange multipliers
//		w += alphaw * dw;
//
//		// Update the y-Lagrange multipliers
//		y += dy;
//
//		// Update the gradient and Hessian at x
//		obj.requires.f = false;
//		obj.requires.g = true;
//		obj.requires.H = true;
//		obj = params.objective(x);
//
//		state.f = obj.f;
//		state.g = obj.g;
//		state.H = obj.H;
//
//		// Return true as found xtrial results in finite f(xtrial)
//		return true;
	};

	/// Return true if the calculation converged.
    auto converged() const -> bool
    {
        // Check if the calculation should stop based on max variation of x
        if(options.tolerancex && max(abs(stepper.step().x)) < options.tolerancex)
            return true;

        // todo you want to make sure the test for convergence should consider the residual of
        // unstable variables in a special way!

        // Check if the calculation should stop based on optimality condititions
        return result.error < options.tolerance;
    };

    auto solve(const OptimumParams& params, OptimumState& state) -> OptimumResult
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

        initialize(params, state);
        outputInitialState(state);

        for(iterations = 1; iterations <= maxiters && !succeeded; ++iterations)
        {
            applyNewtonStepping(params, state);
            outputCurrentState(state);

            if((succeeded = converged()))
                break;

            evaluateObjectiveFunction(params, state);
			computeNewtonStep(params, state);
			updateResultErrors();
        }

        // Output a final header
        outputter.outputHeader();

        // Finish timing the calculation
        result.time = timer.elapsed();

        return result;
    }

    /// Calculate the sensitivity of the optimal solution with respect to parameters.
    auto dxdp(VectorConstRef dgdp, VectorConstRef dbdp) -> Matrix
    {
    	// TODO Implement the calculation of sensitivity derivatives
//        // Initialize the right-hand side of the KKT equations
//        rhs.rx.noalias() = -dgdp;
//        rhs.ry.noalias() =  dbdp;
//        rhs.rz.fill(0.0);
//
//        // Solve the KKT equations to get the derivatives
//        kkt.solve(rhs, sol);

        assert(false);

        // Return the calculated sensitivity vector
        return {};
    }
};

OptimumSolver::OptimumSolver(const OptimumStructure& structure)
: pimpl(new Impl(structure))
{}

OptimumSolver::OptimumSolver(const OptimumSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

OptimumSolver::~OptimumSolver()
{}

auto OptimumSolver::operator=(OptimumSolver other) -> OptimumSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto OptimumSolver::setOptions(const OptimumOptions& options) -> void
{
	pimpl->setOptions(options);
}

auto OptimumSolver::solve(const OptimumParams& params, OptimumState& state) -> OptimumResult
{
    return pimpl->solve(params, state);
}

auto OptimumSolver::dxdp(VectorConstRef dgdp, VectorConstRef dbdp) -> Vector
{
    return pimpl->dxdp(dgdp, dbdp);
}

} // namespace Optima
