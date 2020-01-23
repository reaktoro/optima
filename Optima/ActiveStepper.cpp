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

#include "ActiveStepper.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/SaddlePointMatrix.hpp>
#include <Optima/SaddlePointSolver.hpp>
#include <Optima/Options.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct ActiveStepper::Impl
{
    /// The options for the optimization calculation
    Options options;

    /// The slack vector `z = g + tr(A)*yl + tr(J)*yn`.
    Vector z;

    /// The solution vector `s = [dx dy]`.
    Vector s;

    /// The right-hand side residual vector `r = [rx ry]`.
    Vector r;

    /// The number of variables in x.
    Index n;

    /// The number of free and fixed variables (n = nx + nf).
    Index nx, nf;

    /// The number of stable and unstable (lower/upper) variables among the free variables (nx = ns + nu).
    Index ns, nu;

    /// The number of lower and upper unstable variables (nu = nul + nuu).
    Index nul, nuu;

    // The number of linear and non-linear equality constraints.
    Index ml, mn;

    /// The number of equality constraints (m = ml + mn).
    Index m;

    /// The total number of variables in x and y (t = n + m).
    Index t;

    /// The saddle point solver.
    SaddlePointSolver solver;

    /// The ordering of the variables as (stable, lower unstable, upper unstable, fixed)
    Indices iordering;

    /// Auxiliary vector with lower and upper bounds for all variables,
    /// even those do not actually have bounds (needed to simplify indexing operations!)
    Vector xlower, xupper;

    /// The ordering of the variables with actual lower and upper bounds as (lower/upper unstable, stable)
    Indices ilower, iupper;

    /// The boolean flag that indices if the solver has been initialized
    bool initialized = false;

    /// Construct a default ActiveStepper::Impl instance.
    Impl()
    {}

    /// Initialize the stepper with the structure of the optimization problem.
    auto initialize(const ActiveStepperProblem& problem) -> void
    {
        // Update the initialized status of the solver
        initialized = true;

        // Auxiliary references
        const auto x = problem.x;
        const auto H = problem.H;
        const auto A = problem.A;
        const auto J = problem.J;
        const auto ifixed = problem.ifixed;

        // Initialize the members related to number of variables and constraints
        n  = H.rows();
        ml = A.rows();
        mn = J.rows();
        m  = ml + mn;
        t  = n + m;

        // Initialize the number of fixed and free variables
        nf = ifixed.rows();
        nx = n - nf;

        // Initialize auxiliary vectors
        z      = zeros(n);
        r      = zeros(t);
        s      = zeros(t);
        xlower = constants(n, -infinity());
        xupper = constants(n, +infinity());

        // Initialize the indices of variables with lower/upper bounds removing those with fixed values
        ilower = difference(problem.ilower, ifixed);
        iupper = difference(problem.iupper, ifixed);

        // Initialize the initial ordering of the variables as (free variables, fixed variables)
        iordering = indices(n);
        partitionRight(iordering, ifixed);
    }

    /// Decompose the saddle point matrix for diagonal Hessian matrices.
    auto decompose(const ActiveStepperProblem& problem) -> void
    {
        // Initialize the solver if not yet
        if(!initialized)
            initialize(problem);

        // Auxiliary references
        const auto x = problem.x;
        const auto y = problem.y;
        const auto g = problem.g;
        const auto H = problem.H;
        const auto A = problem.A;
        const auto J = problem.J;

        // The Lagrange multipliers with respect to linear and non-linear constraints
        const auto yl = y.head(ml);
        const auto yn = y.tail(mn);

        // Update the lower/upper bounds of the variables in ilower and iupper
        xlower(problem.ilower) = problem.xlower;
        xupper(problem.iupper) = problem.xupper;

        // Calculate the slack variables z
        z.noalias() = g + tr(A)*yl + tr(J)*yn;

        // Update the ordering of the variables with lower and upper bounds
        auto is_lower_unstable = [&](Index i) { return x[i] == xlower[i] && z[i] > 0.0; };
        auto is_upper_unstable = [&](Index i) { return x[i] == xupper[i] && z[i] < 0.0; };

        // Organize ilower and iupper so that unstable variables are on the beginning
        nul = std::partition(ilower.begin(), ilower.end(), is_lower_unstable) - ilower.begin();
        nuu = std::partition(iupper.begin(), iupper.end(), is_upper_unstable) - iupper.begin();

        // Update the number of unstable and stable variables
        nu = nul + nuu;
        ns = nx - nu;

        // Get the indices of the lower and upper unstable variables
        auto iul = ilower.head(nul);
        auto iuu = iupper.head(nuu);

        // The indices of the free variables
        auto ixx = iordering.head(nx);

        // Move all upper unstable variables to the right among the free variables
        partitionRight(ixx, iuu);

        // Move all lower unstable variables to the right, but before the upper unstable variables
        partitionRight(ixx.head(nx - nuu), iul);

        // The indices of the unstable and fixed variables
        auto iuf = iordering.tail(nu + nf);

        // Setup the saddle point matrix with unstable variables considered "fixed" variables in the problem,
        // together with the actual original fixed variables in `ifixed`
        SaddlePointMatrix spm(H, zeros(n), A, J, iuf);

        // Decompose the saddle point matrix
        solver.decompose(spm);
    }

    /// Solve the saddle point matrix.
    auto solve(const ActiveStepperProblem& problem) -> void
    {
        // Auxiliary references
        const auto x = problem.x;
        const auto y = problem.y;
        const auto g = problem.g;
        const auto h = problem.h;
        const auto A = problem.A;
        const auto J = problem.J;

        // The Lagrange multipliers with respect to linear and non-linear constraints
        const auto yl = y.head(ml);
        const auto yn = y.tail(mn);

        // The indices of the unstable and fixed variables
        auto iuf = iordering.tail(nu + nf);

        // Views to the sub-vectors in `r = [ra rb]`
        auto ra = r.head(n);
        auto rb = r.tail(m);

        // Views to the sub-vectors in `rb = [rbl rbn]`
        auto rbl = rb.head(ml);
        auto rbn = rb.tail(mn);

        // Calculate the right-hand side vector `ra`
        // Note this is similar to the slack variables `z`,
        // which may have changed since `decompose`, say,
        // in a backtracking linear search operation.
        ra.noalias() = -(g + tr(A)*yl + tr(J)*yn);

        // Set entries in `a` coresponding to unstable and fixed variables to zero
        ra(iuf).fill(0.0);

        // Calculate the right-hand side vector `rb = [rbl rbn]`
        rbl.noalias() = -(A*x - problem.b);
        rbn.noalias() = -h;

        // The right-hand side vector `r` of the saddle point problem
        SaddlePointVector rhs(r, n, m);

        // The solution vector of the saddle point problem
        SaddlePointSolution sol(s, n, m);

        // Solve the saddle point problem
        solver.solve(rhs, sol);
    }

    /// Return the calculated Newton step vector.
    auto step() const -> SaddlePointVector
    {
        return SaddlePointVector(s, n, m);
    }

    /// Return the calculated residual vector for the current optimum state.
    auto residual() const -> SaddlePointVector
    {
        return SaddlePointVector(r, n, m);
    }

    /// Return the assembled saddle point matrix.
    auto matrix(const ActiveStepperProblem& problem) -> SaddlePointMatrix
    {
        // Auxiliary references
        const auto H = problem.H;
        const auto A = problem.A;
        const auto J = problem.J;
        const auto iuf = iordering.tail(nu + nf);
        return SaddlePointMatrix(H, zeros(n), A, J, iuf);
    }
};

ActiveStepper::ActiveStepper()
: pimpl(new Impl())
{}

ActiveStepper::ActiveStepper(const ActiveStepper& other)
: pimpl(new Impl(*other.pimpl))
{}

ActiveStepper::~ActiveStepper()
{}

auto ActiveStepper::operator=(ActiveStepper other) -> ActiveStepper&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto ActiveStepper::setOptions(const Options& options) -> void
{
    pimpl->options = options;
    pimpl->solver.setOptions(options.kkt);
}

auto ActiveStepper::decompose(const ActiveStepperProblem& problem) -> void
{
    return pimpl->decompose(problem);
}

auto ActiveStepper::solve(const ActiveStepperProblem& problem) -> void
{
    return pimpl->solve(problem);
}

auto ActiveStepper::matrix(const ActiveStepperProblem& problem) -> SaddlePointMatrix
{
    return pimpl->matrix(problem);
}

auto ActiveStepper::step() const -> SaddlePointVector
{
    return pimpl->step();
}

auto ActiveStepper::residual() const -> SaddlePointVector
{
    return pimpl->residual();
}

} // namespace Optima
