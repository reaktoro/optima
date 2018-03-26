//// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
////
//// Copyright (C) 2014-2018 Allan Leal
////
//// This program is free software: you can redistribute it and/or modify
//// it under the terms of the GNU General Public License as published by
//// the Free Software Foundation, either version 3 of the License, or
//// (at your option) any later version.
////
//// This program is distributed in the hope that it will be useful,
//// but WITHOUT ANY WARRANTY; without even the implied warranty of
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//// GNU General Public License for more details.
////
//// You should have received a copy of the GNU General Public License
//// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
//#include "OptimumSolver.hpp"
//
//// Optima includes
//#include <Optima/Exception.hpp>
//#include <Optima/Outputter.hpp>
//#include <Optima/Timing.hpp>
//#include <Optima/OptimumOptions.hpp>
//#include <Optima/OptimumProblem.hpp>
//#include <Optima/OptimumResult.hpp>
//#include <Optima/OptimumState.hpp>
//#include <Optima/OptimumStepper.hpp>
//#include <Optima/SaddlePointMatrix.hpp>
//#include <Optima/SaddlePointProblem.hpp>
//#include <Optima/Result.hpp>
//#include <Optima/SaddlePointSolver.hpp>
//#include <Optima/Utils.hpp>
//using namespace Eigen;
//
//namespace Optima {
//namespace {
//
//auto isfinite(const ObjectiveState& f) -> bool
//{
//    return std::isfinite(f.val) && f.grad.allFinite();
//}
//
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
//
//} // namespace
//
//struct OptimumSolver::Impl
//{
//    /// The structure of the optimization problem
//    OptimumStructure structure;
//
//    /// The calculator of the Newton step (dx, dy, dz, dw)
//    OptimumStepper stepper;
//
//    /// The evaluation of the objective function.
//    ObjectiveState f;
//
//    /// The trial iterate x
//    Vector xtrial;
//
//    /// The outputter instance
//    Outputter outputter;
//
//    /// The options for the optimization calculation
//    OptimumOptions options;
//
//    /// The number of variables
//    Index n;
//
//    /// The current number of stable, unstable, free, and fixed variables.
//    Index ns, nu, nx, nf;
//
//    /// The number of equality constraints.
//    Index m;
//
//    /// The total number of variables (x, y, z, w).
//    Index t;
//
//    /// Initialize the optimization solver with the structure of the problem.
//    auto initialize(const OptimumStructure& strct) -> void
//    {
//        // Set the structure member with the given one
//        structure = strct;
//
//        // Initialize the stepper solver
//        stepper.initialize(structure);
//
//        // Initialize the members related to number of variables and constraints
//        n = structure.n;
//        m = structure.A.rows();
//
//        // Initialize the number of fixed and free variables in x
//        nf = structure.ifixed().size();
//        nx = n - nf;
//
//        // Allocate memory to x(trial) and H matrix
//        xtrial.resize(n);
//    }
//
//    auto solve(const OptimumParams& params, OptimumState& state) -> OptimumResult
//    {
//        // Start timing the calculation
//        Time begin = time();
//
//        // The result of the calculation
//        OptimumResult result;
//
//        // Finish the calculation if the problem has no variable
//        if(n == 0)
//        {
//            state = {};
//            result.succeeded = true;
//            result.time = elapsed(begin);
//            return result;
//        }
//
//        // Initialize the outputter instance
//        outputter = Outputter();
//        outputter.setOptions(options.output);
//
//        // Set the options for the stepper solver
//        stepper.setOptions(options);
//
//        // Alias to OptimumParams members
//        auto xlower = params.xlower();
//        auto xupper = params.xupper();
//        auto xfixed = params.xfixed();
//
//        // Auxiliary references to general options
//        const auto tol = options.tolerance;
//        const auto tolx = options.tolerancex;
//        const auto maxiters = options.max_iterations;
//
//        // Auxiliary references to some algorithm parameters
//        const auto mu = options.mu;
//        const auto tau = options.tau;
//
//        // Auxiliary references to some result variables
//        auto& error = result.error;
//        auto& iterations = result.iterations;
//        auto& succeeded = result.succeeded = false;
//
//        // The regularization parameters delta and gamma
////        auto gamma = options.regularization.gamma;
////        auto delta = options.regularization.delta;
//
//        // Set gamma and delta to mu in case they are zero
//        // This provides even further regularization to the problem,
//        // as non-zero gamma and delta prevent unbounded primal and dual
//        // variables x and y respectively.
////        gamma = gamma ? gamma : mu;
////        delta = delta ? delta : mu;
//
//        auto& x = state.x;
//        auto& y = state.y;
//        auto& z = state.z;
//        auto& w = state.w;
//
//        // Ensure the initial guesses for `x` and `y` have adequate dimensions
//        if(x.size() != n) x = zeros(n); // original
//        if(y.size() != m) y = zeros(m); // original
//        if(z.size() != n) z = zeros(n); // original
//        if(w.size() != n) w = zeros(n); // original
////        if(x.size() != n) x = ones(n);
////        if(y.size() != m) y = zeros(m);
////        if(z.size() != n) z = constants(n, mu);
//
//        // Ensure the initial guesses for `x` and `z` are inside the feasible domain
//        x.noalias() = (x.array() > 0.0).select(x, mu);
//        z.noalias() = (z.array() > 0.0).select(z, 1.0);
//        w.noalias() = (w.array() < 0.0).select(w, -1.0);
////        x.noalias() = (x.array() > 0.0).select(x, 1.0);
////        z.noalias() = (z.array() > 0.0).select(z, mu);
//        y.noalias() = state.y;
//
//        // The optimality, feasibility, centrality lower and centrality upper
//        double errorf, errorh, errorl, erroru;
//
//        // The function that outputs the header and initial state of the solution
//        auto output_initial_state = [&]()
//        {
//            if(!options.output.active) return;
//
//            outputter.addEntry("Iteration");
//            outputter.addEntry("f(x)");
//            outputter.addEntry("Error");
//            outputter.addEntries(options.output.xprefix, n, options.output.xnames);
//            outputter.addEntries(options.output.yprefix, m, options.output.ynames);
//            outputter.addEntries(options.output.zprefix, n, options.output.znames);
//            outputter.addEntries("r", n, options.output.xnames);
//            outputter.addEntry("Optimality");
//            outputter.addEntry("Feasibility");
//            outputter.addEntry("CentralityLower");
//            outputter.addEntry("CentralityUpper");
//
//            outputter.outputHeader();
//            outputter.addValue(iterations);
//            outputter.addValues(x);
//            outputter.addValues(y);
//            outputter.addValues(z);
//            outputter.addValues(abs(stepper.residualOptimality())); // TODO The ordering of the residual vector needs to be brought to original
//            outputter.addValue(f.val);
//            outputter.addValue(error);
//            outputter.addValue(errorf);
//            outputter.addValue(errorh);
//            outputter.addValue(errorl);
//            outputter.addValue(erroru);
//            outputter.outputState();
//        };
//
//        // The function that outputs the current state of the solution
//        auto output_state = [&]()
//        {
//            if(!options.output.active) return;
//
//            outputter.addValue(iterations);
//            outputter.addValue(f.val);
//            outputter.addValue(error);
//            outputter.addValues(x);
//            outputter.addValues(y);
//            outputter.addValues(z);
//            outputter.addValues(abs(stepper.residualOptimality())); // TODO The ordering of the residual vector needs to be brought to original
//            outputter.addValue(errorf);
//            outputter.addValue(errorh);
//            outputter.addValue(errorl);
//            outputter.addValue(erroru);
//            outputter.outputState();
//        };
//
//        // Return true if the result of a calculation failed
//        auto failed = [&](bool succeeded)
//        {
//            return !succeeded;
//        };
//
//        // The function that computes the current error norms
//        auto update_residuals = [&]()
//        {
//            // todo these errors should probably ignore the residual of unstable variables
//
//            // The indices of the current stable variables
//            auto istable = stepper.istable();
//
//            // Update the optimality, feasibility and centrality errors
//            errorf = norminf(stepper.residualOptimality()(istable));
//            errorh = norminf(stepper.residualFeasibility());
//            errorl = norminf(stepper.residualComplementarityLowerBounds());
//            erroru = norminf(stepper.residualComplementarityUpperBounds());
//            error = std::max({errorf, errorh, errorl, erroru});
//        };
//
//        // The function that initialize the state of some variables
//        auto initialize = [&]()
//        {
//            // Establish the current needs for the objective function evaluation
//            f.requires.val = true;
//            f.requires.grad = true;
//            f.requires.hessian = true;
//
//            // Evaluate the objective function
//            structure.objective(x, f);
//
//            // Use the initial optimization state to assemble and decompose the KKT equations
//            stepper.decompose(params, state, f);
//            stepper.solve(params, state, f);
////            stepper.solve2(params, state, f);
//
//            // Update the residuals of the calculation
//            update_residuals();
//        };
//
//        // The function that computes the Newton step
//        auto compute_newton_step = [&]()
//        {
//            stepper.decompose(params, state, f);
//            stepper.solve(params, state, f);
////            stepper.solve2(params, state, f);
//
//            // Update the time spent in linear systems
////            result.time_linear_systems += res.time();
//
////            // Perform emergency Newton step calculation as long as steps contains NaN or INF values
////            while(!stepper.result().succeeded)
////            {
////                // Increase the value of the regularization parameter delta
////                delta = std::max(delta * 100, 1e-8);
////
////                // Return false if the calculation did not succeeded
////                if(delta > 1e-2) return false;
////
////                // Update the residual of the feasibility conditition
////                rhs.ry -= -delta*delta*y;
////
////                // Update the decomposition of the KKT matrix with update Hessian matrix
////                stepper.decompose(lhs);
////
////                // Compute `dx`, `dy`, `dz` by solving the KKT equation
////                stepper.solve(rhs, sol);
////
////                // Update the time spent in linear systems
////                result.time_linear_systems += stepper.result().time_solve;
////                result.time_linear_systems += stepper.result().time_decompose;
////            }
//
//            // Return true if he calculation succeeded
//            return true;
//        };
//
//        // The aggressive mode for updating the iterates
//        auto update_iterates_aggressive = [&]()
//        {
//            // Aliases to Newton steps
//            auto dx = stepper.dx();
//            auto dy = stepper.dy();
//            auto dz = stepper.dz();
//            auto dw = stepper.dw();
//
//            // Calculate the current trial iterate for x
//            for(int i = 0; i < n; ++i)
//                xtrial[i] = (x[i] + dx[i] > 0.0) ?
//                    x[i] + dx[i] : x[i]*(1.0 - tau);
//
//            // Establish the current needs for the objective function evaluation at the trial iterate
//            f.requires.val = true;
//            f.requires.grad = false;
//            f.requires.hessian = false;
//
//            // Evaluate the objective function at the trial iterate
//            structure.objective(xtrial, f);
//
//            // Initialize the step length factor
//            double alpha = fractionToTheBoundary(x, dx, tau);
//
//            // The number of tentatives to find a trial iterate that results in finite objective result
//            unsigned tentatives = 0;
//
//            // Repeat until f(xtrial) is finite
//            while(!isfinite(f) && ++tentatives < 10)
//            {
//                // Calculate a new trial iterate using a smaller step length
//                xtrial = x + alpha * dx;
//
//                // Evaluate the objective function at the new trial iterate
//                f.requires.val = true;
//                f.requires.grad = false;
//                f.requires.hessian = false;
//                structure.objective(xtrial, f);
//
//                // Decrease the current step length
//                alpha *= 0.5;
//            }
//
//            // Return false if xtrial could not be found s.t. f(xtrial) is finite
//            if(tentatives == 10)
//                return false;
//
//            // Update the iterate x from xtrial
//            x = xtrial;
//
//            // Update the gradient and Hessian at x
//            f.requires.val = false;
//            f.requires.grad = true;
//            f.requires.hessian = true;
//            structure.objective(x, f);
//
//            // Update the z-Lagrange multipliers
//            for(Index i = 0; i < n; ++i)
//                z[i] += (z[i] + dz[i] > 0.0) ?
//                    dz[i] : -tau * z[i];
//
//            // Update the w-Lagrange multipliers
//            for(Index i = 0; i < n; ++i)
//                w[i] += (w[i] + dw[i] > 0.0) ?
//                    dw[i] : -tau * w[i];
//
//            // Update the y-Lagrange multipliers
//            y += dy;
//
//            // Return true as found xtrial results in finite f(xtrial)
//            return true;
//        };
//
//        // The conservative mode for updating the iterates
//        auto update_iterates_convervative = [&]()
//        {
//            // Aliases to Newton steps
//            auto dx = stepper.dx();
//            auto dy = stepper.dy();
//            auto dz = stepper.dz();
//            auto dw = stepper.dw();
//
//            // Initialize the step length factor
//            double alphax = xStepLength(x, dx, xlower, xupper, tau);
//            double alphaz = zStepLength(z, dz, tau);
//            double alphaw = wStepLength(w, dw, tau);
//            double alpha = alphax;
//
//            // The number of tentatives to find a trial iterate that results in finite objective result
//            unsigned tentatives = 0;
//
//            // Repeat until a suitable xtrial iterate if found such that f(xtrial) is finite
//            for(; tentatives < 10; ++tentatives)
//            {
//                // Calculate the current trial iterate for x
//                xtrial = x + alpha * dx;
//
//                // Evaluate the objective function at the trial iterate
//                f.requires.val = true;
//                f.requires.grad = false;
//                f.requires.hessian = false;
//                structure.objective(xtrial, f);
//
//                // Leave the loop if f(xtrial) is finite
//                if(isfinite(f))
//                    break;
//
//                // Decrease alpha in a hope that a shorter step results f(xtrial) finite
//                alpha *= 0.01;
//            }
//
//            // Return false if xtrial could not be found s.t. f(xtrial) is finite
//            if(tentatives == 10)
//                return false;
//
//            // Update the iterate x from xtrial
//            x = xtrial;
//
//            // Update the z-Lagrange multipliers
//            z += alphaz * dz;
//
//            // Update the w-Lagrange multipliers
//            w += alphaw * dw;
//
//            // Update the y-Lagrange multipliers
//            y += dy;
//
//            // Update the gradient and Hessian at x
//            f.requires.val = false;
//            f.requires.grad = true;
//            f.requires.hessian = true;
//            structure.objective(x, f);
//
//            // Return true as found xtrial results in finite f(xtrial)
//            return true;
//        };
//
//        // The function that performs an update in the iterates
//        auto update_iterates = [&]()
//        {
//            switch(options.step)
//            {
//            case Aggressive: return update_iterates_aggressive();
//            default: return update_iterates_convervative();
//            }
//        };
//
//        auto converged = [&]()
//        {
//            // Check if the calculation should stop based on max variation of x
//            if(tolx && max(abs(stepper.dx())) < tolx)
//                return true;
//
//            // todo you want to make sure the test for convergence should consider the residual of
//            // unstable variables in a special way!
//
//            // Check if the calculation should stop based on optimality condititions
//            return error < tol;
//        };
//
//        initialize();
//        output_initial_state();
//
//        for(iterations = 1; iterations <= maxiters && !succeeded; ++iterations)
//        {
//            if(failed(compute_newton_step()))
//                break;
//            if(failed(update_iterates()))
//                break;
//            if((succeeded = converged()))
//                break;
//            update_residuals();
//            output_state();
//        }
//
//        // Output a final header
//        outputter.outputHeader();
//
//        // Finish timing the calculation
//        result.time = elapsed(begin);
//
//        return result;
//    }
//
//    /// Solve the optimization problem.
////    auto solve(const OptimumProblem& problem, OptimumState& state) -> OptimumResult
////    {
////        // Start timing the calculation
////        Time begin = time();
////
////        // The result of the calculation
////        OptimumResult result;
////
////        // Finish the calculation if the problem has no variable
////        if(problem.n == 0)
////        {
////            state = OptimumState();
////            result.succeeded = true;
////            result.time = elapsed(begin);
////            return result;
////        }
////
////        // Initialize the outputter instance
////        outputter = Outputter();
////        outputter.setOptions(options.output);
////
////        // Set the KKT options
////        kkt.setOptions(options.kkt);
////
////        // Define some auxiliary references to variables
////        auto& x = state.x;
////        auto& y = state.y;
////        auto& z = state.z;
////        auto& f = state.f;
////
////        // The number of variables and equality constraints
////        const auto& A = problem.A;
////        const auto& a = problem.a;
////        const auto& n = problem.A.cols();
////        const auto& m = problem.A.rows();
////
////        // Define auxiliary references to general options
////        const auto tol = options.tolerance;
////        const auto tolx = options.tolerancex;
////        const auto maxiters = options.max_iterations;
////
////        // Define some auxiliary references to IpNewton parameters
////        const auto mu = options.ipnewton.mu;
////        const auto tau = options.ipnewton.tau;
////
////        // Define some auxiliary references to result variables
////        auto& error = result.error;
////        auto& iterations = result.iterations;
////        auto& succeeded = result.succeeded = false;
////
////        // The regularization parameters delta and gamma
////        auto gamma = options.regularization.gamma;
////        auto delta = options.regularization.delta;
////
////        // Set gamma and delta to mu in case they are zero
////        // This provides even further regularization to the problem,
////        // as non-zero gamma and delta prevent unbounded primal and dual
////        // variables x and y respectively.
////        gamma = gamma ? gamma : mu;
////        delta = delta ? delta : mu;
////
////        // Ensure the initial guesses for `x` and `y` have adequate dimensions
////        if(x.size() != n) x = zeros(n);
////        if(y.size() != m) y = zeros(m);
////        if(z.size() != n) z = zeros(n);
////
////        // Ensure the initial guesses for `x` and `z` are inside the feasible domain
////        x = (x.array() > 0.0).select(x, mu);
////        z = (z.array() > 0.0).select(z, 1.0);
////
////        // The transpose representation of matrix `A`
////        const auto At = tr(A);
////
////        // The KKT matrix
////        KktMatrix lhs(f.hessian, A, x, z, gamma, delta);
////
////        // The optimality, feasibility, centrality and total error variables
////        double errorf, errorh, errorc;
////
////        // The function that outputs the header and initial state of the solution
////        auto output_initial_state = [&]()
////        {
////            if(!options.output.active) return;
////
////            outputter.addEntry("Iteration");
////            outputter.addEntries(options.output.xprefix, n, options.output.xnames);
////            outputter.addEntries(options.output.yprefix, m, options.output.ynames);
////            outputter.addEntries(options.output.zprefix, n, options.output.znames);
////            outputter.addEntries("r", n, options.output.xnames);
////            outputter.addEntry("f(x)");
////            outputter.addEntry("Error");
////            outputter.addEntry("Optimality");
////            outputter.addEntry("Feasibility");
////            outputter.addEntry("Centrality");
////
////            outputter.outputHeader();
////            outputter.addValue(iterations);
////            outputter.addValues(x);
////            outputter.addValues(y);
////            outputter.addValues(z);
////            outputter.addValues(abs(rhs.rx));
////            outputter.addValue(f.val);
////            outputter.addValue(error);
////            outputter.addValue(errorf);
////            outputter.addValue(errorh);
////            outputter.addValue(errorc);
////            outputter.outputState();
////        };
////
////        // The function that outputs the current state of the solution
////        auto output_state = [&]()
////        {
////            if(!options.output.active) return;
////
////            outputter.addValue(iterations);
////            outputter.addValues(x);
////            outputter.addValues(y);
////            outputter.addValues(z);
////            outputter.addValues(abs(rhs.rx));
////            outputter.addValue(f.val);
////            outputter.addValue(error);
////            outputter.addValue(errorf);
////            outputter.addValue(errorh);
////            outputter.addValue(errorc);
////            outputter.outputState();
////        };
////
////        // Return true if the result of a calculation failed
////        auto failed = [&](bool succeeded)
////        {
////            return !succeeded;
////        };
////
////        // The function that computes the current error norms
////        auto update_residuals = [&]()
////        {
////            // Compute the right-hand side vectors of the KKT equation
////            rhs.rx.noalias() = -(f.grad - At*y - z + gamma*gamma*ones(n));
////            rhs.ry.noalias() = -(A*x + delta*delta*y - a);
////            rhs.rz.noalias() = -(x % z - mu);
////
////            // Calculate the optimality, feasibility and centrality errors
////            errorf = norminf(rhs.rx);
////            errorh = norminf(rhs.ry);
////            errorc = norminf(rhs.rz);
////            error = std::max({errorf, errorh, errorc});
////        };
////
////        // The function that initialize the state of some variables
////        auto initialize = [&]()
////        {
////            // Initialize xtrial
////            xtrial.resize(n);
////
////            // Evaluate the objective function
////            f.requires = {};
////            structure.objective(x, f);
////
////            // Update the residuals of the calculation
////            update_residuals();
////        };
////
////        // The function that computes the Newton step
////        auto compute_newton_step = [&]()
////        {
////            // Update the decomposition of the KKT matrix with update Hessian matrix
////            kkt.decompose(lhs);
////
////            // Compute `dx`, `dy`, `dz` by solving the KKT equation
////            kkt.solve(rhs, sol);
////
////            // Update the time spent in linear systems
////            result.time_linear_systems += kkt.result().time_solve;
////            result.time_linear_systems += kkt.result().time_decompose;
////
////            // Perform emergency Newton step calculation as long as steps contains NaN or INF values
////            while(!kkt.result().succeeded)
////            {
////                // Increase the value of the regularization parameter delta
////                delta = std::max(delta * 100, 1e-8);
////
////                // Return false if the calculation did not succeeded
////                if(delta > 1e-2) return false;
////
////                // Update the residual of the feasibility conditition
////                rhs.ry -= -delta*delta*y;
////
////                // Update the decomposition of the KKT matrix with update Hessian matrix
////                kkt.decompose(lhs);
////
////                // Compute `dx`, `dy`, `dz` by solving the KKT equation
////                kkt.solve(rhs, sol);
////
////                // Update the time spent in linear systems
////                result.time_linear_systems += kkt.result().time_solve;
////                result.time_linear_systems += kkt.result().time_decompose;
////            }
////
////            // Return true if he calculation succeeded
////            return true;
////        };
////
////        // The aggressive mode for updating the iterates
////        auto update_iterates_aggressive = [&]()
////        {
////            // Calculate the current trial iterate for x
////            for(int i = 0; i < n; ++i)
////                xtrial[i] = (x[i] + dx[i] > 0.0) ?
////                    x[i] + dx[i] : x[i]*(1.0 - tau);
////
////            // Evaluate the objective function at the trial iterate
////            f.requires.val = true;
////            f.requires.grad = false;
////            f.requires.hessian = false;
////            structure.objective(xtrial, f);
////
////            // Initialize the step length factor
////            double alpha = fractionToTheBoundary(x, dx, tau);
////
////            // The number of tentatives to find a trial iterate that results in finite objective result
////            unsigned tentatives = 0;
////
////            // Repeat until f(xtrial) is finite
////            while(!isfinite(f) && ++tentatives < 10)
////            {
////                // Calculate a new trial iterate using a smaller step length
////                xtrial = x + alpha * dx;
////
////                // Evaluate the objective function at the trial iterate
////                f.requires.val = true;
////              f.requires.grad = false;
////              f.requires.hessian = false;
////                structure.objective(xtrial, f);
////
////                // Decrease the current step length
////                alpha *= 0.5;
////            }
////
////            // Return false if xtrial could not be found s.t. f(xtrial) is finite
////            if(tentatives == 10)
////                return false;
////
////            // Update the iterate x from xtrial
////            x = xtrial;
////
////            // Update the gradient and Hessian at x
////            f.requires.val = false;
////          f.requires.grad = true;
////          f.requires.hessian = true;
////            structure.objective(x, f);
////
////            // Update the z-Lagrange multipliers
////            for(int i = 0; i < n; ++i)
////                z[i] += (z[i] + dz[i] > 0.0) ?
////                    dz[i] : -tau * z[i];
////
////            // Update the y-Lagrange multipliers
////            y += dy;
////
////            // Return true as found xtrial results in finite f(xtrial)
////            return true;
////        };
////
////        // The conservative mode for updating the iterates
////        auto update_iterates_convervative = [&]()
////        {
////            // Initialize the step length factor
////            double alphax = fractionToTheBoundary(x, dx, tau);
////            double alphaz = fractionToTheBoundary(z, dz, tau);
////            double alpha = alphax;
////
////            // The number of tentatives to find a trial iterate that results in finite objective result
////            unsigned tentatives = 0;
////
////            // Repeat until a suitable xtrial iterate if found such that f(xtrial) is finite
////            for(; tentatives < 10; ++tentatives)
////            {
////                // Calculate the current trial iterate for x
////                xtrial = x + alpha * dx;
////
////                // Evaluate the objective function at the trial iterate
////                f.requires.val = true;
////              f.requires.grad = false;
////              f.requires.hessian = false;
////                structure.objective(xtrial, f);
////
////                // Leave the loop if f(xtrial) is finite
////                if(isfinite(f))
////                    break;
////
////                // Decrease alpha in a hope that a shorter step results f(xtrial) finite
////                alpha *= 0.01;
////            }
////
////            // Return false if xtrial could not be found s.t. f(xtrial) is finite
////            if(tentatives == 10)
////                return false;
////
////            // Update the iterate x from xtrial
////            x = xtrial;
////
////            // Update the z-Lagrange multipliers
////            z += alphaz * dz;
////
////            // Update the y-Lagrange multipliers
////            y += dy;
////
////            // Update the gradient and Hessian at x
////            f.requires.val = false;
////          f.requires.grad = true;
////          f.requires.hessian = true;
////            structure.objective(x, f);
////
////            // Return true as found xtrial results in finite f(xtrial)
////            return true;
////        };
////
////        // The function that performs an update in the iterates
////        auto update_iterates = [&]()
////        {
////            switch(options.ipnewton.step)
////            {
////            case Aggressive: return update_iterates_aggressive();
////            default: return update_iterates_convervative();
////            }
////        };
////
////        auto converged = [&]()
////        {
////            // Check if the calculation should stop based on max variation of x
////            if(tolx && max(abs(dx)) < tolx)
////                return true;
////
////            // Check if the calculation should stop based on optimality condititions
////            return error < tol;
////        };
////
////        initialize();
////        output_initial_state();
////
////        for(iterations = 1; iterations <= maxiters && !succeeded; ++iterations)
////        {
////            if(failed(compute_newton_step()))
////                break;
////            if(failed(update_iterates()))
////                break;
////            if((succeeded = converged()))
////                break;
////            update_residuals();
////            output_state();
////        }
////
////        // Output a final header
////        outputter.outputHeader();
////
////        // Finish timing the calculation
////        result.time = elapsed(begin);
////
////        return result;
////    }
////
////    /// Calculate the sensitivity of the optimal solution with respect to parameters.
////    auto dxdp(VectorConstRef dgdp, VectorConstRef dbdp) -> Matrix
////    {
////        // Initialize the right-hand side of the KKT equations
////        rhs.rx.noalias() = -dgdp;
////        rhs.ry.noalias() =  dbdp;
////        rhs.rz.fill(0.0);
////
////        // Solve the KKT equations to get the derivatives
////        kkt.solve(rhs, sol);
////
////        // Return the calculated sensitivity vector
////        return dx;
////    }
//
//    /// Calculate the sensitivity of the optimal solution with respect to parameters.
//    auto dxdp(VectorConstRef dgdp, VectorConstRef dbdp) -> Matrix
//    {
////        // Initialize the right-hand side of the KKT equations
////        rhs.rx.noalias() = -dgdp;
////        rhs.ry.noalias() =  dbdp;
////        rhs.rz.fill(0.0);
////
////        // Solve the KKT equations to get the derivatives
////        kkt.solve(rhs, sol);
//
//        assert(false);
//
//        // Return the calculated sensitivity vector
//        return stepper.dx();
//    }
//};
//
//OptimumSolver::OptimumSolver()
//: pimpl(new Impl())
//{}
//
//OptimumSolver::OptimumSolver(const OptimumSolver& other)
//: pimpl(new Impl(*other.pimpl))
//{}
//
//OptimumSolver::~OptimumSolver()
//{}
//
//auto OptimumSolver::operator=(OptimumSolver other) -> OptimumSolver&
//{
//    pimpl = std::move(other.pimpl);
//    return *this;
//}
//
//auto OptimumSolver::setOptions(const OptimumOptions& options) -> void
//{
//    pimpl->options = options;
//}
//
//auto OptimumSolver::initialize(const OptimumStructure& structure) -> void
//{
//    pimpl->initialize(structure);
//}
//
//auto OptimumSolver::solve(const OptimumParams& params, OptimumState& state) -> OptimumResult
//{
//    return pimpl->solve(params, state);
//}
//
//auto OptimumSolver::solve(const OptimumProblem& problem, OptimumState& state) -> OptimumResult
//{
//    pimpl->initialize(problem);
//    return pimpl->solve(problem, state);
//}
//
//auto OptimumSolver::dxdp(VectorConstRef dgdp, VectorConstRef dbdp) -> Vector
//{
//    return pimpl->dxdp(dgdp, dbdp);
//}
//
//} // namespace Optima
