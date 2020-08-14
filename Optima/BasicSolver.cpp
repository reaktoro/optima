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
// along with this program. If not, see <http://www.fxnu.org/licenses/>.

#include "BasicSolver.hpp"

// C++ includes
#include <cassert>
#include <vector>

// Optima includes
#include <Optima/Analysis.hpp>
#include <Optima/Exception.hpp>
#include <Optima/Options.hpp>
#include <Optima/Outputter.hpp>
#include <Optima/Result.hpp>
#include <Optima/Stepper.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {
namespace {

const auto FAILED    = false; ///< The boolean flag that indicates a step failed.
const auto SUCCEEDED = true;  ///< The boolean flag that indicates a step succeeded.

/// Return true if `x` is not finite.
inline auto isinf(double x) -> bool { return !std::isfinite(x); }

} // anonymous namespace

/// The implementation of the solver for basic optimization problems.
struct BasicSolver::Impl
{
    //======================================================================
    // DATA INITIALIZED AT CONSTRUCTION TIME
    //======================================================================

    Stepper stepper; ///< The calculator of the Newton step (dx, dy).
    Matrix Ax;       ///< The constant coefficient matrix A in the linear equality constraints in Ax*x + Ap*p = b.
    Matrix Ap;       ///< The constant coefficient matrix A in the linear equality constraints in Ax*x + Ap*p = b.
    Index nx;        ///< The number of primal variables x.
    Index np;        ///< The number of parameter variables p.
    Index mb;        ///< The number of linear equality constraints in Ax*x + Ap*p = b.
    Index mh;        ///< The number of non-linear equality constraints in h(x, p) = 0.
    Index m;         ///< The number of constraints in Ax*x + Ap*p = b and h(x, p) = 0.
    Index t;         ///< The total number of variables in (x, p, y).

    //======================================================================
    // DATA INITIALIZED AT THE BEGINNING OF EACH SOLVE OPERATION
    //======================================================================

    ObjectiveFunction objectivefn;   ///< The objective function *f(x, p)* of the basic optimization problem.
    ConstraintFunction constraintfn; ///< The nonlinear equality constraint function *h(x, p)*.
    ConstraintFunction vfn;          ///< The external nonlinear constraint function *v(x, p)*.
    Vector b;                        ///< The right-hand side vector *b* of the linear equality constraints *Ax*x + Ap*p = b*.
    Vector xlower;                   ///< The lower bounds of the primal variables *x*.
    Vector xupper;                   ///< The upper bounds of the primal variables *x*.
    Vector plower;                   ///< The lower bounds of the parameter variables *p*.
    Vector pupper;                   ///< The upper bounds of the parameter variables *p*.
    Stability stability;             ///< The stability state of the primal variables *x*.

    //======================================================================
    // DATA THAT IS CHANGED IN EVERY ITERATION OF THE ALGORITHM
    //======================================================================

    Result result;     ///< The result of the optimization problem.
    Analysis analysis; ///< The data for convergence analysis of the optimization calculation.
    double f;          ///< The evaluated objective function f(x, p).
    Vector fx;         ///< The evaluated gradient of the objective function f(x, p) with respect to x.
    Matrix fxx;        ///< The evaluated Jacobian of the gradient function fx(x, p) with respect to x, i.e., the Hessian of f(x, p).
    Matrix fxp;        ///< The evaluated Jacobian of the gradient function fx(x, p) with respect to p.
    Vector h;          ///< The evaluated equality constraint function h(x, p).
    Matrix hx;         ///< The evaluated Jacobian of the equality constraint function h(x, p) with respect to x.
    Matrix hp;         ///< The evaluated Jacobian of the equality constraint function h(x, p) with respect to p.
    Vector v;          ///< The evaluated external constraint function v(x, p).
    Matrix vx;         ///< The evaluated Jacobian of the external constraint function v(x, p) with respect to x.
    Matrix vp;         ///< The evaluated Jacobian of the external constraint function v(x, p) with respect to p.
    Vector dx;         ///< The Newton step for the primal variables *x*.
    Vector dp;         ///< The Newton step for the parameter variables *p*.
    Vector dy;         ///< The Newton step for the Lagrange multipliers *y*.
    Vector rx;         ///< The residuals of the first-order optimality conditions.
    Vector rp;         ///< The residuals of the external constraint functions *v(x, p)*.
    Vector ry;         ///< The residuals of the linear/nonlinear feasibility conditions.
    Vector ex;         ///< The relative errors of the first-order optimality conditions.
    Vector ep;         ///< The relative errors of the external constraint functions *v(x, p)*.
    Vector ey;         ///< The relative errors of the linear/nonlinear feasibility conditions.
    Vector x;          ///< The current value of x.
    Vector p;          ///< The current value of p.
    Vector y;          ///< The current value of y.
    Vector z;          ///< The current value of z = g + tr(Ax)yl + tr(hx)*yn.
    Vector xtrial;     ///< The trial iterate x(trial).
    Vector ptrial;     ///< The trial iterate p(trial).
    Vector ytrial;     ///< The trial iterate y(trial).
    Vector xbar;       ///< The auxiliary vector x used during the line-search algorithm.
    Vector pbar;       ///< The auxiliary vector p used during the line-search algorithm.
    Vector ybar;       ///< The auxiliary vector y used during the line-search algorithm.
    Vector dxtrial;    ///< The trial Newton step dx(trial).
    Vector dptrial;    ///< The trial Newton step dp(trial).
    Vector dytrial;    ///< The trial Newton step dy(trial).
    double L;          ///< The current value L(x, p, y) of the Lagrange function.
    double E;          ///< The current error E(x, p, y) = ||grad(L)||^2.

    //======================================================================
    // OTHER DATA
    //======================================================================

    Options options;     ///< The options for the optimization calculation.
    Outputter outputter; ///< The outputter object to output computation state.

    /// Construct a BasicSolver::Impl instance with given details of the optimization problem.
    Impl(BasicSolverInitArgs args)
    : stepper({ args.nx, args.np, args.m, args.Ax, args.Ap }), Ax(args.Ax), Ap(args.Ap)
    {
        // Initialize the members related to number of variables and constraints
        nx = args.nx;
        np = args.np;
        m  = args.m;
        mb = args.Ax.rows();
        mh = m - mb;
        t  = nx + np + m;

        // Ensure consistent dimensions of vectors/matrices.
        assert(nx > 0);
        assert(mb <= m);

        // Allocate memory
        fx      = zeros(nx);
        fxx     = zeros(nx, nx);
        fxp     = zeros(nx, np);
        b       = zeros(mb);
        h       = zeros(mh);
        hx      = zeros(mh, nx);
        hp      = zeros(mh, np);
        v       = zeros(np);
        vx      = zeros(np, nx);
        vp      = zeros(np, np);
        dx      = zeros(nx);
        dp      = zeros(np);
        dy      = zeros(m);
        rx      = zeros(nx);
        rp      = zeros(np);
        ry      = zeros(m);
        ex      = zeros(nx);
        ep      = zeros(np);
        ey      = zeros(m);
        x       = zeros(nx);
        p       = zeros(np);
        y       = zeros(m);
        z       = zeros(nx);
        xtrial  = zeros(nx);
        ptrial  = zeros(np);
        ytrial  = zeros(m);
        xbar    = zeros(nx);
        pbar    = zeros(np);
        ybar    = zeros(m);
        dxtrial = zeros(nx);
        dptrial = zeros(np);
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
        if(nx == 0 && np == 0)
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

        auto [fxw, hw, bw, vw, stability, xw, pw, yw, zw] = args;

        stepper.sensitivities({ fxw, hw, bw, vw, stability, xw, pw, yw, zw });

        result.time_sensitivities = timer.elapsed();

        return result;
    }

    // Output the header at the top of the output table.
    auto outputHeaderTop() -> void
    {
        if(!options.output.active) return;
        outputter.addEntry("Iteration");
        outputter.addEntry("f(x, p)");
        outputter.addEntry("L(x, p, y)");
        outputter.addEntry("E(x, p, y)");
        outputter.addEntry("Optimality");
        outputter.addEntry("Feasibility");
        outputter.addEntries(options.output.xprefix, nx, options.output.xnames);
        outputter.addEntries(options.output.pprefix, np, options.output.pnames);
        outputter.addEntries(options.output.yprefix, m, options.output.ynames);
        outputter.addEntries(options.output.zprefix, nx, options.output.xnames);
        outputter.addEntries("rx", nx, options.output.xnames);
        outputter.addEntries("rp", np, options.output.pnames);
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
        outputter.addValues(p);
        outputter.addValues(y);
        outputter.addValues(z);
        outputter.addValues(abs(rx));
        outputter.addValues(abs(rp));
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
        assert(args.x.rows() == nx);
        assert(args.p.rows() == np);
        assert(args.y.rows() == m);
        assert(args.b.rows() == mb);
        assert(args.xlower.rows() == nx);
        assert(args.xupper.rows() == nx);
        assert(args.plower.rows() == np);
        assert(args.pupper.rows() == np);

        // Ensure functions h and v are given if mh > 0 and np > 0 respectively
        assert(mh == 0 || args.h);
        assert(np == 0 || args.v);

        // Clear previous state of the Outputter instance
        outputter.clear();

        // Initialize the problem related data-members
        objectivefn  = args.obj;
        constraintfn = args.h;
        vfn          = args.v;
        b            = args.b;
        xlower       = args.xlower;
        xupper       = args.xupper;
        plower       = args.plower;
        pupper       = args.pupper;

        // Initialize vectors x, p, y with given initial guesses
        x = args.x;
        p = args.p;
        y = args.y;

        // Initialize the stability of the variables with given initial state
        stability = args.stability;

        // Ensure the initial guesses for `x` and `p` do not violate their lower/upper bounds
        x.noalias() = min(max(args.x, xlower), xupper);
        p.noalias() = min(max(args.p, plower), pupper);

    	// Initialize the Newton step calculator once for the upcoming decompose/solve calls
        stepper.initialize({ b, xlower, xupper, plower, pupper, x, stability });

        // Initialize the convergence analysis data of the optimization calculation
        analysis.initialize(options.max_iterations);

        // Evaluate the objective function f and its gradient g at x0 (initial guess)
        const auto fres = evaluateObjectiveFn(x, p, { .f=true, .fx=true, .fxx=true }); // TODO: The Hessian computation at this point should be eliminated in the future. Check how this can be done safely.

        // Return false if objective function evaluation failed.
        if(fres.failed) return FAILED;

        // Evaluate the equality constraint function h at x0 (initial guess)
        const auto hres = evaluateConstraintFn(x, p);

        // Return false if constraint function evaluation failed.
        if(hres.failed) return FAILED;

        // Evaluate the external constraint function v at x0 (initial guess)
        const auto vres = evaluateExternalConstraintFn(x, p);

        // Return false if external constraint function evaluation failed.
        if(vres.failed) return FAILED;

        // Canonicalize the Ax matrix as a pre-step to calculate the Newton step
        stepper.canonicalize({ x, p, y, fx, fxx, fxp, vx, vp, hx, hp, xlower, xupper, plower, pupper, stability });

        // Return true as initialize step was successful.
        return SUCCEEDED;
	}

    // Finalize the calculation by setting back computed state.
    auto finalize(BasicSolverSolveArgs args) -> void
	{
        // Set back in args the solution state of variables x, p, y, z
        args.x = x;
        args.p = p;
        args.y = y;
        args.z = z;

        // Set back in args the obtained stability state of the variables
        args.stability = stability;
    }

    // Evaluate the objective function.
    auto evaluateObjectiveFn(const Vector& x, const Vector& p, ObjectiveRequirement req) -> ObjectiveResult
	{
        // Start time measuring.
    	Timer timer;

        // Create an ObjectiveResult with f, fx, fxx to be filled
        ObjectiveResult res(f, fx, fxx, fxp);

        // The computation requirements for f, g, and H
        res.requires = req;

        // Evaluate the objective function f(x, p)
        objectivefn(x, p, res);

        // Check the objective function produces finite numbers at this point
        if(res.requires.f && isinf(f))
            res.failed = true;
        if(res.requires.fx && !fx.allFinite())
            res.failed = true;
        if(res.requires.fxx && !fxx.allFinite())
            res.failed = true;

        // Update the number of objective function calls
        result.num_objective_evals   += 1;
        result.num_objective_evals_f += res.requires.f;
        result.num_objective_evals_g += res.requires.fx;
        result.num_objective_evals_H += res.requires.fxx;

        // Update the time spent on objective function evaluation
        result.time_objective_evals += timer.elapsed();

        return res;
	}

    // Evaluate the equality constraint function h(x, p)
    auto evaluateConstraintFn(const Vector& x, const Vector& p) -> ConstraintResult
	{
        // Start time measuring.
    	Timer timer;

        // Create a ConstraintResult with h, hx, hp to be filled
        ConstraintResult res{h, hx, hp};

        // Skip if there are no non-linear equality constraints
        if(mh == 0)
            return res;

        // Evaluate the constraint function h(x, p)
        constraintfn(x, p, res);

        // Check the constraint function produces finite numbers at this point
        if(!h.allFinite() || !hx.allFinite() || !hp.allFinite())
            res.failed = true;

        // Update the time spent on constraint function evaluation
        result.time_constraint_evals += timer.elapsed();

        return res;
	}

    // Evaluate the external constraint function v(x, p)
    auto evaluateExternalConstraintFn(const Vector& x, const Vector& p) -> ConstraintResult
	{
        // Start time measuring.
    	Timer timer;

        // Create a ConstraintResult with v, vx, vp to be filled
        ConstraintResult res{v, vx, vp};

        // Skip if there are no p-variables
        if(np == 0)
            return res;

        // Evaluate the constraint function v(x, p)
        vfn(x, p, res);

        // Check the external constraint function produces finite numbers at this point
        if(!v.allFinite() || !vx.allFinite() || !vp.allFinite())
            res.failed = true;

        // Update the time spent on constraint function evaluation
        result.time_constraint_evals += timer.elapsed();

        return res;
	}

	// Update the optimality, feasibility and complementarity errors
	auto updateResiduals() -> void
	{
        // Compute the current optimality and feasibility residuals (rx, ry) and relative errors (ex, ey)
        stepper.residuals({ x, p, y, b, h, v, fx, hx, rx, rp, ry, ex, ep, ey, z });

		// Update the current optimality, feasibility and complementarity errors
		result.error_optimality  = norminf(ex);
		result.error_feasibility = norminf(ey);

		// Update the current maximum error of the optimization calculation
		result.error = std::max({
			result.error_optimality,
			result.error_feasibility
		});

        // Update the Lagrange function L(x, p, y) = f + tr(yb)*(Ax*x + Ap*p - b) + tr(yh)*h
        const auto yb = y.head(mb);
        const auto yh = y.tail(mh);

        L = f + yb.dot(Ax*x + Ap*p - b) + yh.dot(h);

        // Update the error E(x, p, y) = ||g + tr(Ax)yb + tr(hx)yh||^2 + ||Ax*x + Ap*p - b||^2 + ||h(x, p)||^2 + ||v(x, p)||^2.
        stepper.steepestDescentLagrange({ x, p, y, fx, b, h, v, dx, dp, dy });
        E = rx.squaredNorm() + rp.squaredNorm() + ry.squaredNorm();

        // Store both L and E in the analysis container
        analysis.L.push_back(L);
        analysis.E.push_back(E);
	}

	// Compute the Lagrange function L(x, p, y) used in the line search algorithm.
	auto computeLagrangeFn(const Vector& x, const Vector& p, const Vector& y) -> double
    {
        // Evaluate the objective function f(x, p) and its gradient fx(x, p)
        const auto fres = evaluateObjectiveFn(x, p, { .f=true, .fx=true, .fxx=false });

        // Return +inf if objective function evaluation failed.
        if(fres.failed) return infinity();

        // Evaluate the equality constraint function h(x, p)
        const auto hres = evaluateConstraintFn(x, p);

        // Return +inf if constraint function evaluation failed.
        if(hres.failed) return infinity();

        // Update the Lagrange function L(x, p, y) = f + tr(yb)*(Ax*x + Ap*p - b) + tr(yh)*h
        const auto yb = y.head(mb);
        const auto yh = y.tail(mh);

        L = f + yb.dot(Ax*x + Ap*p - b) + yh.dot(h);

        return L;
    }

	// Compute the error function E(x, p, y) used in the line search algorithm.
	auto computeError(const Vector& x, const Vector& p, const Vector& y) -> double
    {
        // Evaluate the objective function gradient fx(x, p) only
        const auto fres = evaluateObjectiveFn(x, p, { .f=false, .fx=true, .fxx=false });

        // Return +inf if objective function evaluation failed.
        if(fres.failed) return infinity();

        // Evaluate the equality constraint function h(x, p)
        const auto hres = evaluateConstraintFn(x, p);

        // Return +inf if constraint function evaluation failed.
        if(hres.failed) return infinity();

        // Compute the current optimality and feasibility residuals.
        // This can be achieved with Stepper::steepestDescentLagrange method.
        stepper.steepestDescentLagrange({ x, p, y, fx, b, h, v, dx, dp, dy });

		// Return the error E(x, p, y) = ||g + tr(W)y||^2 + ||Ax*x + Ap*p - b||^2 + ||h(x, p)||^2 + ||v(x, p)||^2.
		return rx.squaredNorm() + rp.squaredNorm() + ry.squaredNorm();
    }

    // The function that computes the Newton step
    auto computeNewtonStep() -> bool
    {
        // Start time measuring.
    	Timer timer;

        // Evaluate only the Hessian of the objective function, since the gradient has already been evaluated.
        const auto fres = evaluateObjectiveFn(x, p, { .f=false, .fx=false, .fxx=true });

        // Ensure the Hessian computation was successul.
        if(fres.failed)
            return FAILED;

        // Canonicalize the W = [A; J] matrix as a pre-step to calculate the Newton step
        stepper.canonicalize({ x, p, y, fx, fxx, fxp, vx, vp, hx, hp, xlower, xupper, plower, pupper, stability });

    	// Decompose the Jacobian matrix and calculate a Newton step
        stepper.decompose({ x, p, y, fx, fxx, fxp, vx, vp, hx, hp, xlower, xupper, plower, pupper, stability });

        // Calculate the Newton step
        stepper.solve({ x, p, y, fx, b, h, v, stability, dx, dp, dy });

        // Update the time spent in linear systems
		result.time_linear_systems += timer.elapsed();

        // Newton step was calculated successfully.
        return SUCCEEDED;
    };

	// Update the variables (x, p, y) with a Newton step.
	auto applyNewtonStep() -> bool
    {
        // Compute the Newton steps dx, dp, dy
        if(computeNewtonStep() == FAILED)
            return FAILED;

        // Compute x(trial), p(trial), y(trial) taking care of the bounds of x and p
        xtrial = x;
        ptrial = p;
        performAggressiveStep(xtrial, dx, xlower, xupper);
        performAggressiveStep(ptrial, dp, plower, pupper);
        ytrial = y + dy;

        // Determine if line-search operations are needed, based on errors at x(trial), p(trial), y(trial).
        if(isLineSearchNeeded(xtrial, ptrial, ytrial))
            if(initiateLineSearch(xtrial, ptrial, ytrial) == FAILED)
                return FAILED;

        // Update x, p, y to their respective trial states
        x = xtrial;
        p = ptrial;
        y = ytrial;

        // The Newton step approach was successful.
        return SUCCEEDED;
    }

    /// Return true if current trial state for x, p, y demands a line-search procedure.
    auto isLineSearchNeeded(const Vector& xtrial, const Vector& ptrial, const Vector& ytrial) -> bool
    {
        // Compute the new error E after Newton step approach
        const auto Enew = computeError(xtrial, ptrial, ytrial);

        // The error E at the initial guess
        const auto E0 = analysis.E.front();

        // The error E at the previous iteration
        const auto Eold = analysis.E.back();

        //======================================================================
        // Return true if Enew == inf (i.e., evaluation of f(x, p) or h(x, p) failed!).
        //======================================================================
        if(isinf(Enew))
            return true;

        //======================================================================
        // Return true if Enew is much larger than E0
        //======================================================================
        {
            const auto factor = options.linesearch.trigger_when_current_error_is_greater_than_initial_error_by_factor;
            if(factor > 0.0 && Enew > factor * E0)
                return true;
        }

        //======================================================================
        // Return true if Enew is much larger than Eold
        //======================================================================
        {
            const auto factor = options.linesearch.trigger_when_current_error_is_greater_than_previous_error_by_factor;
            if(factor > 0.0 && Enew > factor * Eold)
                return true;
        }

        return false;
    }

    /// Perform a line-search along the computed Newton direction.
    auto initiateLineSearch(Vector& xtrial, Vector& ptrial, Vector& ytrial) -> bool
    {
        // Start with x(bar) = x(current), p(bar) = p(current), y(bar) = y(current)
        xbar = x;
        pbar = p;
        ybar = y;

        // Step x(bar) until the full-length of dx if no bounds are violated.
        // Otherwise, stop at the first hit lower/upper bound of x.
        const auto maxlength = performConservativeStep(xbar, dx, xlower, xupper);

        // Step p(bar) taking care of the bounds of p
        performAggressiveStep(pbar, dp, plower, pupper);

        // Update y(bar) considering the length obtained in the previous step for x.
        ybar += maxlength*dy;

        // Compute the new error E after conservative Newton step approach
        const auto Ebar = computeError(xbar, pbar, ybar);

        // Check if Ebar is infinity. This condition is a result of a failure
        // in the evaluation of the objective and constraint functions f(x, p) and
        // h(x, p). This backtrack step procedure will compute x(bar), p(bar) y(bar)
        // that does not cause failures in the evaluation of such functions.
        if(isinf(Ebar))
            if(initiateBacktrackStepping(xbar, pbar, ybar) == FAILED)
                return FAILED;

        // Define the function phi(alpha) = E(x + alpha*dx) that we want to minimize.
        const auto phi = [&](double alpha) -> double
        {
            xtrial.noalias() = x*(1 - alpha) + alpha*xbar; // using x + alpha*(xbar - x) is sensitive to round-off errors!
            ptrial.noalias() = p*(1 - alpha) + alpha*pbar; // using p + alpha*(pbar - p) is sensitive to round-off errors!
            ytrial.noalias() = y*(1 - alpha) + alpha*ybar; // using y + alpha*(ybar - y) is sensitive to round-off errors!
            return computeError(xtrial, ptrial, ytrial);
        };

        // The tolerance and maximum number of iterations used in the line-search minimization procedure.
        const auto tol = options.linesearch.tolerance;
        const auto maxiters = options.linesearch.maxiters;

        // Minimize function phi(alpha) along the computed Newton direction `dx` and `dy`.
        // This is to be performed in the interval [0, 1], where alpha=1
        // coincides with the largest Newton step that we could make so that no
        // lower/upper bound is violated.
        const auto alphamin = minimizeBrent(phi, 0.0, 1.0, tol, maxiters);

        // Calculate x(trial), p(trial), y(trial) using the minimizer alpha value
        xtrial.noalias() = x*(1 - alphamin) + alphamin*xbar; // using x + alpha*(xbar - x) is sensitive to round-off errors!
        ptrial.noalias() = p*(1 - alphamin) + alphamin*pbar; // using p + alpha*(pbar - p) is sensitive to round-off errors!
        ytrial.noalias() = y*(1 - alphamin) + alphamin*ybar; // using y + alpha*(ybar - y) is sensitive to round-off errors!

        // Compute the new error E after the line-search operation.
        const auto Enew = computeError(xtrial, ptrial, ytrial);

        // Return failed status only if the new error is infinity (i.e. when
        // the evaluation of the objective/constraint functions fail at the
        // last alpha). Note the error found here may still be larger than the
        // previous error, or even the error at the initial guess. The
        // line-search goal is only to tentatively tame large errors resulting
        // from a plain use of full-length Newton step (or a step that
        // immediately brings x[i] to its bounds).
        if(isinf(Enew))
            return FAILED;

        // The Newton step approach was successful.
        return SUCCEEDED;
    }

	// Perform shorter and shorter Newton steps until objective function does not fail.
	auto initiateBacktrackStepping(Vector& xbar, Vector& pbar, Vector& ybar) -> bool
    {
        // The parameter used to decrease the Newton steps during the backtrack search
        const auto factor = options.backtrack.factor;

        // The maximum number of tentatives in case of failure when applying Newton steps
        const auto maxiters = options.backtrack.maxiters;

        // The alpha length to be determined in the backtrack search operation below
        auto alpha = factor;

        for(auto i = 0; i < maxiters; ++i)
        {
            xtrial.noalias() = x*(1 - alpha) + alpha*xbar; // using x + alpha*(xbar - x) is sensitive to round-off errors!
            ptrial.noalias() = p*(1 - alpha) + alpha*pbar; // using p + alpha*(pbar - p) is sensitive to round-off errors!
            ytrial.noalias() = y*(1 - alpha) + alpha*ybar; // using y + alpha*(ybar - y) is sensitive to round-off errors!

            const auto Enew = computeError(xtrial, ptrial, ytrial);

            if(!isinf(Enew))
            {
		        xbar = xtrial;
		        pbar = ptrial;
		        ybar = ytrial;
                return SUCCEEDED;
            }

            alpha *= factor;
        }

        return FAILED;
    }

	// Update the variables (x, p, y) with a steepest descent step.
	auto applySteepestDescentStep() -> bool
    {
        // Evaluate the Hessian of the objective function
        const auto fres = evaluateObjectiveFn(x, p, { .f=false, .fx=true, .fxx=true });

        // Ensure the Hessian computation was successul.
        if(fres.failed)
            return FAILED;

        // Compute the steepest descent steps dx, dp, dy
        stepper.steepestDescentError({ x, p, y, fx, fxx, fxp, b, h, hx, hp, v, vx, vp, dx, dp, dy });

        // Start with x(bar) = x(current), p(bar) = p(current), y(bar) = y(current)
        xbar = x;
        pbar = p;
        ybar = y;

        // Step x(bar) until the full-length of `dx` if no bounds are violated.
        // Otherwise, stop at the first hit lower/upper bound.
        const auto maxlength = performConservativeStep(xbar, dx, xlower, xupper);

        // Step p(bar) taking care of the bounds of p
        performAggressiveStep(pbar, dp, plower, pupper);

        // Update y(bar) considering the length obtained in the previous step.
        ybar += maxlength*dy;

        // Compute the Lagrange value L at x(bar), p(bar) y(bar) after a full
        // steepest descent step (or tamed to avoid bound violation!).
        const auto Ebar = computeError(xbar, pbar, ybar);

        // The Lagrange value L at the previous iteration.
        const auto Eold = analysis.E.back();

        // Skip the minimization procedure if the Lagrange function has already
        // been decreased with the full steepest descent step. This is needed
        // because the minimization approach would be more costly to find this
        // out. Also, the minimization approach below may produce an alpha
        // length that is close to 1, but not 1. This can produce an indefinite
        // number of movements towards a bound, with the variable closest to
        // its bound never getting attached to it.
        if(Ebar < Eold)
        {
            x = xbar;
            p = pbar;
            y = ybar;
            return SUCCEEDED;
        }

        // Define the function phi(alpha) = L(x + alpha*dx, p + alpha*dp, y + alpha*dy) that we want to minimize.
        const auto phi = [&](double alpha) -> double
        {
            xtrial.noalias() = x*(1 - alpha) + alpha*xbar; // using x + alpha*(xbar - x) is sensitive to round-off errors!
            ptrial.noalias() = p*(1 - alpha) + alpha*pbar; // using p + alpha*(pbar - p) is sensitive to round-off errors!
            ytrial.noalias() = y*(1 - alpha) + alpha*ybar; // using y + alpha*(ybar - y) is sensitive to round-off errors!
            return computeError(xtrial, ptrial, ytrial);
        };

        // The tolerance and maximum number of iterations used in the steepest descent minimization procedure.
        const auto tol = options.steepestdescent.tolerance;
        const auto maxiters = options.steepestdescent.maxiters;

        // Minimize function phi(alpha) along the computed steepest descent direction `dx` and `dy`.
        // This is to be performed in the interval [0, 1], where alpha=1
        // coincides with the largest Newton step that we could make so that no
        // lower/upper bound is violated.
        const auto alphamin = minimizeBrent(phi, 0.0, 1.0, tol, maxiters);

        // Calculate x(trial), p(trial), y(trial) using the minimizer alpha value
        xtrial.noalias() = x*(1 - alphamin) + alphamin*xbar; // using x + alpha*(xbar - x) is sensitive to round-off errors!
        ptrial.noalias() = p*(1 - alphamin) + alphamin*pbar; // using p + alpha*(pbar - p) is sensitive to round-off errors!
        ytrial.noalias() = y*(1 - alphamin) + alphamin*ybar; // using y + alpha*(ybar - y) is sensitive to round-off errors!

        // Compute the new Lagrange value L after the steepest descent operation.
        const auto Enew = computeError(xtrial, ptrial, ytrial);

        // Return failed status if the Lagrange function did not decrease along
        // the steepest descent minimization operation.
        if(Enew > Eold)
            return FAILED;

        // Update x, p, y to their respective trial states
        x = xtrial;
        p = ptrial;
        y = ytrial;

        // The steepest descent step approach was successful.
        return SUCCEEDED;
    }

	// Update the variables (x, p, y) with a steepest descent step.
	auto applySteepestDescentLagrangeStep() -> bool
    {
        // Compute the steepest descent steps dx, dp, dy
        stepper.steepestDescentLagrange({ x, p, y, fx, b, h, v, dx, dp, dy });

        // Start with x(bar) = x(current), p(bar) = p(current), y(bar) = y(current)
        xbar = x;
        pbar = p;
        ybar = y;

        // Step x(bar) until the full-length of `dx` if no bounds are violated.
        // Otherwise, stop at the first hit lower/upper bound.
        const auto maxlength = performConservativeStep(xbar, dx, xlower, xupper);

        // Step p(bar) taking care of the bounds of p
        performAggressiveStep(pbar, dp, plower, pupper);

        // Update y(bar) considering the length obtained in the previous step.
        ybar += maxlength*dy;

        // Compute the Lagrange value L at x(bar), p(bar) y(bar) after a full
        // steepest descent step (or tamed to avoid bound violation!).
        const auto Lbar = computeLagrangeFn(xbar, pbar, ybar);

        // The Lagrange value L at the previous iteration.
        const auto Lold = analysis.L.back();

        // Skip the minimization procedure if the Lagrange function has already
        // been decreased with the full steepest descent step. This is needed
        // because the minimization approach would be more costly to find this
        // out. Also, the minimization approach below may produce an alpha
        // length that is close to 1, but not 1. This can produce an indefinite
        // number of movements towards a bound, with the variable closest to
        // its bound never getting attached to it.
        if(Lbar < Lold)
            return SUCCEEDED;

        // Define the function phi(alpha) = L(x + alpha*dx, p + alpha*dp, y + alpha*dy) that we want to minimize.
        const auto phi = [&](double alpha) -> double
        {
            xtrial.noalias() = x*(1 - alpha) + alpha*xbar; // using x + alpha*(xbar - x) is sensitive to round-off errors!
            ytrial.noalias() = y*(1 - alpha) + alpha*ybar; // using y + alpha*(ybar - y) is sensitive to round-off errors!
            return computeLagrangeFn(xtrial, ptrial, ytrial);
        };

        // The tolerance and maximum number of iterations used in the steepest descent minimization procedure.
        const auto tol = options.steepestdescent.tolerance;
        const auto maxiters = options.steepestdescent.maxiters;

        // Minimize function phi(alpha) along the computed steepest descent direction `dx` and `dy`.
        // This is to be performed in the interval [0, 1], where alpha=1
        // coincides with the largest Newton step that we could make so that no
        // lower/upper bound is violated.
        const auto alphamin = minimizeBrent(phi, 0.0, 1.0, tol, maxiters);

        // Calculate x(trial), p(trial), y(trial) using the minimizer alpha value
        xtrial.noalias() = x*(1 - alphamin) + alphamin*xbar; // using x + alpha*(xbar - x) is sensitive to round-off errors!
        ptrial.noalias() = p*(1 - alphamin) + alphamin*pbar; // using p + alpha*(pbar - p) is sensitive to round-off errors!
        ytrial.noalias() = y*(1 - alphamin) + alphamin*ybar; // using y + alpha*(ybar - y) is sensitive to round-off errors!

        // Compute the new Lagrange value L after the steepest descent operation.
        const auto Lnew = computeLagrangeFn(xtrial, ptrial, ytrial);

        // Return failed status if the Lagrange function did not decrease along
        // the steepest descent minimization operation.
        if(Lnew > Lold)
            return FAILED;

        // The steepest descent step approach was successful.
        return SUCCEEDED;
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
