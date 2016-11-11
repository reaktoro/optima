// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2016 Allan Leal
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

// C++ includes
#include <algorithm>

// Optima includes
#include <Optima/Common/Exception.hpp>
#include <Optima/Core/OptimumMethod.hpp>
#include <Optima/Core/OptimumOptions.hpp>
#include <Optima/Core/OptimumProblem.hpp>
#include <Optima/Core/OptimumResult.hpp>
#include <Optima/Core/OptimumSolverIpNewton.hpp>
#include <Optima/Core/OptimumState.hpp>
#include <Optima/Core/Regularizer.hpp>
#include <Optima/Math/LU.hpp>
#include <Optima/Math/Utils.hpp>

namespace Optima {

struct OptimumSolver::Impl
{
    /// The pointer to the optimization solver
    OptimumSolverBase* solver = nullptr;

    /// The regularized optimization problem (i.e., no linearly dependent equality constraints and removed trivial constraints)
    OptimumProblem rproblem;

    /// The regularized optimization options
    OptimumOptions roptions;

    /// The regularizer of the linear equality constraints
    Regularizer regularizer;

    // Construct a default Impl instance
    Impl()
    {
        // Set the IpNewton method as default
        setMethod(OptimumMethod::IpNewton);
    }

    // Construct a Impl instance with given method
    Impl(OptimumMethod method)
    {
        setMethod(method);
    }

    ~Impl()
    {
        if(solver != nullptr) delete solver;
    }

    // Set the optimization method for the solver
    auto setMethod(OptimumMethod method) -> void
    {
        if(solver != nullptr) delete solver;

        switch(method)
        {
        case OptimumMethod::IpNewton:
            solver = new OptimumSolverIpNewton();
            break;
        default:
            solver = new OptimumSolverIpNewton();
            break;
        }
    }

    // Solve the optimization problem
    auto solve(const OptimumProblem& problem, OptimumState& state, const OptimumOptions& options) -> OptimumResult
    {
        // Assert either objective function or c vector have been initialized
        if(problem.objective && problem.c.size())
            RuntimeError("Could not solve the optimization problem.",
                "The given OptimumProblem instance is ambiguous: "
                    "both members `c` and `objective` have been initialized.");

        // The number of variables, constraints, and sensitivity parameters
        const auto m = problem.A.rows();
        const auto n = problem.A.cols();

        // Assert the number of columns in A is compatible with number of variables
        Assert(static_cast<Index>(n) == problem.n,
            "Could not solve the optimization problem.",
            "The constraint matrix A has " + std::to_string(n) + " columns, but the "
            "optimization problem has " + std::to_string(problem.n) + " variables.");

        // Ensure x, y, z have correct dimensions
        if(state.x.size() != n) state.x = zeros(n);
        if(state.y.size() != m) state.y = zeros(m);
        if(state.z.size() != n) state.z = zeros(n);

        // Initialize the regularized problem, state, and options instances
        rproblem = problem;
        roptions = options;

        // Update the options of the regularizer
        regularizer.setOptions(options.regularization);

        // Regularize the equality constraints
        regularizer.regularize(rproblem, state, roptions);

        // Solve the regularized problem
        OptimumResult result = solver->solve(rproblem, state, roptions);

        // Recover the regularized solution to the one corresponding to original problem
        regularizer.recover(state);

        return result;
    }

    /// Calculate the sensitivity of the optimal solution with respect to parameters.
    auto dxdp(Vector dgdp, Vector dbdp) -> Vector
    {
        // Assert the size of the input matrices dgdp and dbdp
        Assert(dgdp.rows() && dbdp.rows() && dgdp.cols() == dbdp.cols(),
            "Could not calculate the sensitivity of the optimal solution with respect to parameters.",
            "The given input matrices `dgdp` and `dbdp` are either empty or does not have the same number of columns.");

        // Check if the last regularized problem had only trivial variables
        if(rproblem.n == 0)
            return zeros(dgdp.rows());

        // Regularize dg/dp and db/dp by removing trivial components, linearly dependent components, etc.
        regularizer.regularize(dgdp, dbdp);

        // Compute the sensitivity dx/dp of x with respect to p
        Vector dxdp = solver->dxdp(dgdp, dbdp);

        // Recover `dx/dp` in case there are trivial variables
        regularizer.recover(dxdp);

        return dxdp;
    }
};

OptimumSolver::OptimumSolver()
: pimpl(new Impl())
{}

OptimumSolver::OptimumSolver(OptimumMethod method)
: pimpl(new Impl(method))
{}

OptimumSolver::OptimumSolver(const OptimumSolver& other)
: pimpl(new Impl(*other.pimpl))
{
    // Clone the optimization solver from the copying object
    pimpl->solver = other.pimpl->solver->clone();
}

OptimumSolver::~OptimumSolver()
{}

auto OptimumSolver::operator=(OptimumSolver other) -> OptimumSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto OptimumSolver::setMethod(OptimumMethod method) -> void
{
    pimpl->setMethod(method);
}

auto OptimumSolver::approximate(const OptimumProblem& problem, OptimumState& state) -> OptimumResult
{
    return approximate(problem, state, {});
}

auto OptimumSolver::approximate(const OptimumProblem& problem, OptimumState& state, const OptimumOptions& options) -> OptimumResult
{
	RuntimeError("Not implemented.", "Not implemented.");
}

auto OptimumSolver::solve(const OptimumProblem& problem, OptimumState& state) -> OptimumResult
{
    return solve(problem, state, {});
}

auto OptimumSolver::solve(const OptimumProblem& problem, OptimumState& state, const OptimumOptions& options) -> OptimumResult
{
    return pimpl->solve(problem, state, options);
}

auto OptimumSolver::dxdp(const Vector& dgdp, const Vector& dbdp) -> Vector
{
    return pimpl->dxdp(dgdp, dbdp);
}

} // namespace Optima
