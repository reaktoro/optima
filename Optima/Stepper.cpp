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

#include "Stepper.hpp"

// C++ includes
#include <cassert>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/SaddlePointSolver.hpp>
#include <Optima/Options.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct Stepper::Impl
{
    Options options;          ///< The options for the optimization calculation
    Matrix W;                 ///< The coefficient matrix W = [A; J] of the linear/nonlinear equality constraints.
    Vector z;                 ///< The instability measures of the variables defined as `z = g + tr(W)*y`.
    Vector s;                 ///< The solution vector `s = [dx dy]` for the saddle point problem.
    Vector r;                 ///< The right-hand side residual vector `r = [rx ry]`  for the saddle point problem.
    Index n      = 0;         ///< The number of variables in x.
    Index ns     = 0;         ///< The number of stable variables in x.
    Index nu     = 0;         ///< The number of unstable (lower/upper) variables in x.
    Index nlower = 0;         ///< The number of variables with lower bounds.
    Index nupper = 0;         ///< The number of variables with upper bounds.
    Index ml     = 0;         ///< The number of linear equality constraints.
    Index mn     = 0;         ///< The number of non-linear equality constraints.
    Index m      = 0;         ///< The number of equality constraints (m = ml + mn).
    Index t      = 0;         ///< The total number of variables in x and y (t = n + m).
    SaddlePointSolver solver; ///< The saddle point solver.
    Indices iordering;        ///< The ordering of the variables as (*stable*, *lower unstable*, *upper unstable*).
    Indices ilowerpartition;  ///< The ordering of the variables as (with lower bounds, without lower bounds, fixed)
    Indices iupperpartition;  ///< The ordering of the variables as (with upper bounds, without upper bounds, fixed)

    /// Construct a default Stepper::Impl instance.
    Impl()
    {}

    /// Construct a Stepper::Impl instance.
    Impl(StepperInitArgs args)
    : n(args.n), m(args.m), W(args.m, args.n), solver({args.n, args.m, args.A})
    {
        // Ensure the step calculator is initialized with a positive number of variables.
        Assert(n > 0, "Could not proceed with Stepper initialization.",
            "The number of variables is zero.");

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

        // Initialize the initial ordering of the variables
        iordering = indices(n);
    }

    /// Initialize the step calculator before calling decompose multiple times.
    auto initialize(StepperInitializeArgs args) -> void
    {
        // Auxiliary references
        auto xlower = args.xlower;
        auto xupper = args.xupper;

        // Ensure consistent dimensions of vectors/matrices.
        assert(xlower.rows() == n);
        assert(xupper.rows() == n);
        assert(iordering.rows() == n);

        // Update the ordering of the variables with lower and upper bounds
        auto has_lower_bound = [&](Index i) -> bool { return !std::isinf(xlower[i]); };
        auto has_upper_bound = [&](Index i) -> bool { return !std::isinf(xupper[i]); };

        ilowerpartition = iordering;
        iupperpartition = iordering;

        nlower = moveLeftIf(ilowerpartition, has_lower_bound);
        nupper = moveLeftIf(iupperpartition, has_upper_bound);
    }

    /// Decompose the saddle point matrix for diagonal Hessian matrices.
    auto decompose(StepperDecomposeArgs args) -> void
    {
        // Ensure the step calculator has been initialized.
        Assert(n != 0, "Could not proceed with Stepper::decompose.",
            "Stepper object not initialized.");

        // Auxiliary references
        auto x          = args.x;
        auto y          = args.y;
        auto g          = args.g;
        auto H          = args.H;
        auto J          = args.J;
        auto xlower     = args.xlower;
        auto xupper     = args.xupper;
        auto ivariables = args.iordering;
        auto& nul       = args.nul;
        auto& nuu       = args.nuu;
        const auto A    = W.topRows(ml);

        // Ensure consistent dimensions of vectors/matrices.
        assert(x.rows() == n);
        assert(y.rows() == m);
        assert(g.rows() == n);
        assert(H.rows() == n);
        assert(H.cols() == n);
        assert(J.rows() == mn);
        assert(J.cols() == n || mn == 0);
        assert(xlower.rows() == n);
        assert(xupper.rows() == n);
        assert(ivariables.rows() == n);
        assert(iordering.rows() == n);
        assert(W.rows() == m);
        assert(W.cols() == n || m == 0);

        // Update the coefficient matrix W = [A; J] with the updated J block
        W.bottomRows(mn) = J;

        // Calculate the optimality residuals
        z.noalias() = g + tr(W)*y;

        // The indices of the variables with lower bounds (ilower) and upper bounds (iupper)
        auto ilower = ilowerpartition.head(nlower);
        auto iupper = iupperpartition.head(nupper);

        // Update the ordering of the variables with lower and upper bounds
        auto is_lower_unstable = [&](Index i) { return x[i] == xlower[i] && z[i] > 0.0; };
        auto is_upper_unstable = [&](Index i) { return x[i] == xupper[i] && z[i] < 0.0; };

        // Organize ilower and iupper as (unstable, stable)
        nul = moveLeftIf(ilower, is_lower_unstable);
        nuu = moveLeftIf(iupper, is_upper_unstable);

        // Update the number of unstable and stable variables
        nu = nul + nuu;
        ns = n - nu;

        // The indices of the lower and upper unstable variables
        // Remember: ilower and iupper are organized in the order [unstable variables, stable variables]!
        auto iul = ilower.head(nul);
        auto iuu = iupper.head(nuu);

        // Move all upper unstable variables to the right in iordering
        moveIntersectionRight(iordering, iuu);

        // Move all lower unstable variables to the right in iordering, but before the unstable variables
        moveIntersectionRight(iordering.head(n - nuu), iul);

        // The indices of the lower and upper unstable variables
        auto iu = iordering.tail(nu);

        // Decompose the saddle point matrix (this decomposition is later used in method solve, possibly many times!)
        // Consider lower/upper unstable variables as "fixed" variables in the saddle point problem.
        // Reason: we do not need to compute Newton steps for the currently unstable variables!
        solver.decompose({ H, J, Matrix{}, iu });

        // Export the updated ordering of the variables
        ivariables = iordering;
    }

    /// Solve the saddle point problem.
    auto solve(StepperSolveArgs args) -> void
    {
        // Auxiliary references
        auto x       = args.x;
        auto y       = args.y;
        auto b       = args.b;
        auto g       = args.g;
        auto h       = args.h;
        auto dx      = args.dx;
        auto dy      = args.dy;
        auto rx      = args.rx;
        auto ry      = args.ry;
        auto z       = args.z;
        const auto A = W.topRows(ml);

        // Ensure consistent dimensions of vectors/matrices.
        assert(x.rows() == n);
        assert(y.rows() == m);
        assert(g.rows() == n);
        assert(A.rows() == ml);
        assert(A.cols() == n || ml == 0);
        assert(iordering.rows() == n);

        // The indices of the lower and upper unstable variables
        auto iu = iordering.tail(nu);

        // Calculate the instability measure of the variables.
        z.noalias() = g + tr(W)*y;

        // Calculate the residuals of the first-order optimality conditions
        rx = -z;

        // Set residuals wrt unstable variables to zero
        rx(iu).fill(0.0);

        // Calculate the residuals of the feasibility conditions
        ry.head(ml).noalias() = -(A*x - b);
        ry.tail(mn).noalias() = -h;

        // Solve the saddle point problem
        solver.solve({ rx, ry, dx, dy });
    }

    /// Compute the sensitivity derivatives of the saddle point problem.
    auto sensitivities(StepperSensitivitiesArgs args) -> void
    {
        // Auxiliary references
        auto dxdp = args.dxdp;
        auto dydp = args.dydp;
        auto dzdp = args.dzdp;
        auto dgdp = args.dgdp;
        auto dbdp = args.dbdp;
        auto dhdp = args.dhdp;

        // The number of parameters for sensitivity computation
        auto np = dxdp.cols();

        // Ensure consistent dimensions of vectors/matrices.
        assert(dxdp.rows() == n);
        assert(dydp.rows() == m);
        assert(dzdp.rows() == n);
        assert(dgdp.rows() == n);
        assert(dbdp.rows() == ml);
        assert(dhdp.rows() == mn);
        assert(dydp.cols() == np);
        assert(dzdp.cols() == np);
        assert(dgdp.cols() == np);
        assert(dbdp.cols() == np);
        assert(dhdp.cols() == np);

        // The indices of the stable and unstable variables
        auto is = iordering.head(ns);
        auto iu = iordering.tail(nu);

        // Assemble the right-hand side matrix (zero for unstable variables!)
        dxdp(is, all) = -dgdp(is, all);
        dxdp(iu, all).fill(0.0);

        // Assemble the right-hand side matrix with dbdp and -dhdp contributions
        dydp.topRows(ml) = dbdp;
        dydp.bottomRows(mn) = -dhdp;

        // Solve the saddle point problem for each parameter to compute the corresponding sensitivity derivatives
        for(Index i = 0; i < np; ++i)
            solver.solve({ dxdp.col(i), dydp.col(i) });

        // Calculate the sensitivity derivatives dzdp (zero for stable variables!).
        dzdp(is, all).fill(0.0);
        dzdp(iu, all) = dgdp(iu, all) + tr(W(all, iu)) * dydp;
    }
};

Stepper::Stepper()
: pimpl(new Impl())
{}

Stepper::Stepper(StepperInitArgs args)
: pimpl(new Impl(args))
{}

Stepper::Stepper(const Stepper& other)
: pimpl(new Impl(*other.pimpl))
{}

Stepper::~Stepper()
{}

auto Stepper::operator=(Stepper other) -> Stepper&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Stepper::setOptions(const Options& options) -> void
{
    pimpl->options = options;
    pimpl->solver.setOptions(options.kkt);
}

auto Stepper::initialize(StepperInitializeArgs args) -> void
{
    return pimpl->initialize(args);
}

auto Stepper::decompose(StepperDecomposeArgs args) -> void
{
    return pimpl->decompose(args);
}

auto Stepper::solve(StepperSolveArgs args) -> void
{
    return pimpl->solve(args);
}

auto Stepper::sensitivities(StepperSensitivitiesArgs args) -> void
{
    return pimpl->sensitivities(args);
}

} // namespace Optima
