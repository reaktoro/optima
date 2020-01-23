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

    /// The coefficient matrix W = [A; J] of the linear/nonlinear equality constraints.
    Matrix W;

    /// The instability measures of the variables defined as `z = g + tr(W)*y`.
    Vector z;

    /// The solution vector `s = [dx dy]` for the saddle point problem.
    Vector s;

    /// The right-hand side residual vector `r = [rx ry]`  for the saddle point problem.
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

    /// Auxiliary vector with lower and upper bounds for all variables, even those do not actually have bounds (needed to simplify indexing operations!)
    Vector xlower, xupper;

    /// The ordering of the variables with actual lower and upper bounds as (lower/upper unstable, stable)
    Indices ilower, iupper;

    /// The assembled saddle point matrix.
    Matrix M;

    /// Construct a default ActiveStepper::Impl instance.
    Impl()
    {}

    /// Construct a ActiveStepper::Impl instance.
    Impl(const ActiveStepperInitArgs& args)
    : n(args.n), m(args.m), W(args.m, args.n)
    {
        // Initialize number of fixed and free variables
        nf = args.ifixed.rows();
        nx = n - nf;

        // Initialize number of linear and nonlinear equality constraints
        ml = args.A.rows();
        mn = m - ml;

        // Initialize the matrix W = [A; J], with J=0 at this initialization time (updated at each decompose call)
        W << args.A, zeros(mn, n);

        // Initialize total number of variables x and y
        t  = n + m;

        // Initialize auxiliary vectors
        z = zeros(n);
        r = zeros(t);
        s = zeros(t);

        // Initialize the indices of variables with lower/upper bounds removing those with fixed values
        ilower = difference(args.ilower, args.ifixed);
        iupper = difference(args.iupper, args.ifixed);

        // Initialize the lower bounds of the variables in ilower
        xlower = constants(n, -infinity());
        xlower(args.ilower) = args.xlower;

        // Initialize the upper bounds of the variables in ilower
        xupper = constants(n, +infinity());
        xupper(args.iupper) = args.xupper;

        // Initialize the initial ordering of the variables as (free variables, fixed variables)
        iordering = indices(n);
        partitionRight(iordering, args.ifixed);
    }

    /// Decompose the saddle point matrix for diagonal Hessian matrices.
    auto decompose(const ActiveStepperDecomposeArgs& args) -> void
    {
        // Auxiliary references
        const auto x = args.x;
        const auto y = args.y;
        const auto g = args.g;
        const auto H = args.H;
        const auto J = args.J;
        const auto A = W.topRows(ml);

        // Update the coefficient matrix W = [A; J] with the updated J block
        W.bottomRows(mn) = J;

        // Calculate the optimality residuals
        z.noalias() = g + tr(W)*y;

        // Update the ordering of the variables with lower and upper bounds
        auto is_lower_unstable = [&](Index i) { return x[i] == xlower[i] && z[i] > 0.0; };
        auto is_upper_unstable = [&](Index i) { return x[i] == xupper[i] && z[i] < 0.0; };

        // Organize ilower and iupper so that unstable variables are on the beginning
        nul = std::partition(ilower.begin(), ilower.end(), is_lower_unstable) - ilower.begin();
        nuu = std::partition(iupper.begin(), iupper.end(), is_upper_unstable) - iupper.begin();

        // Update the number of unstable and stable variables
        nu = nul + nuu;
        ns = nx - nu;

        // The indices of the lower and upper unstable variables
        // Remember: ilower and iupper are organized in the order [unstable variables, stable variables]!
        auto iul = ilower.head(nul);
        auto iuu = iupper.head(nuu);

        // The indices of the free variables
        // Remember: iordering is organized in the order [free variables, fixed variables]!
        auto ixx = iordering.head(nx);

        // Move all upper unstable variables to the right among the free variables
        partitionRight(ixx, iuu);

        // Move all lower unstable variables to the right, but before the upper unstable variables
        partitionRight(ixx.head(nx - nuu), iul);

        // The indices of the unstable and fixed variables
        auto iuf = iordering.tail(nu + nf);

        // Setup the saddle point matrix.
        // Consider lower/upper unstable variables as "fixed" variables in the saddle point problem.
        // These are in addition to the original fixed variables in `ifixed`.
        // Reason: we do not need to compute Newton steps for the currently unstable and fixed variables!
        SaddlePointMatrix spm(H, zeros(n), A, J, iuf);

        // Decompose the saddle point matrix (this decomposition is later used in method solve, possibly many times!)
        solver.decompose(spm);
    }

    /// Solve the saddle point problem.
    auto solve(const ActiveStepperSolveArgs& args, ActiveStepperSolution sol) -> void
    {
        // Auxiliary references
        const auto x = args.x;
        const auto y = args.y;
        const auto b = args.b;
        const auto g = args.g;
        const auto h = args.h;
        const auto A = W.topRows(ml);

        // The indices of the unstable and fixed variables
        auto iuf = iordering.tail(nu + nf);

        // Calculate the instability measure of the variables.
        sol.z.noalias() = g + tr(W)*y;

        // Calculate the residuals of the first-order optimality conditions
        sol.rx = -sol.z;
        sol.rx(iuf).fill(0.0); // unstable and fixed variables have zero right-hand side values!

        // Calculate the residuals of the feasibility conditions
        sol.ry.head(ml).noalias() = -(A*x - b);
        sol.ry.tail(mn).noalias() = -h;

        // Solve the saddle point problem
        solver.solve({sol.rx, sol.ry}, {sol.dx, sol.dy});
    }

    /// Return the assembled saddle point matrix.
    auto matrix(const ActiveStepperDecomposeArgs& args) -> SaddlePointMatrix
    {
        decompose(args);
        const auto H = args.H;
        const auto J = args.J;
        const auto A = W.topRows(ml);
        const auto iuf = iordering.tail(nu + nf);
        return SaddlePointMatrix(H, zeros(n), A, J, iuf);
    }
};

ActiveStepper::ActiveStepper()
: pimpl(new Impl())
{}

ActiveStepper::ActiveStepper(const ActiveStepperInitArgs& args)
: pimpl(new Impl(args))
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

auto ActiveStepper::decompose(const ActiveStepperDecomposeArgs& args) -> void
{
    return pimpl->decompose(args);
}

auto ActiveStepper::solve(const ActiveStepperSolveArgs& args, ActiveStepperSolution sol) -> void
{
    return pimpl->solve(args, sol);
}

auto ActiveStepper::matrix(const ActiveStepperDecomposeArgs& args) -> SaddlePointMatrix
{
    return pimpl->matrix(args);
}

} // namespace Optima
