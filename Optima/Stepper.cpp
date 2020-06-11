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

using std::abs;

struct Stepper::Impl
{
    Options options;          ///< The options for the optimization calculation
    Matrix W;                 ///< The coefficient matrix W = [A; J] of the linear/nonlinear equality constraints.
    Vector z;                 ///< The instability measures of the variables defined as z = g + tr(W)*y.
    Vector sx;                ///< The solution vector x in the saddle point problem.
    Vector sy;                ///< The solution vector y in the saddle point problem.
    Vector sa;                ///< The right-hand side vector a in the saddle point problem.
    Vector sb;                ///< The right-hand side vector b in the saddle point problem.
    Vector Hx;                ///< The product of Hessian matrix *H* and the current state of the primal variables *x*
    Index n  = 0;             ///< The number of variables in x.
    Index ml = 0;             ///< The number of linear equality constraints.
    Index mn = 0;             ///< The number of non-linear equality constraints.
    Index m  = 0;             ///< The number of equality constraints (m = ml + mn).
    Index t  = 0;             ///< The total number of variables in x and y (t = n + m).
    Indices ipositiverows;    ///< The indices of the rows in matrix A that only have positive or zero coefficients.
    Indices inegativerows;    ///< The indices of the rows in matrix A that only have negative or zero coefficients.
    SaddlePointSolver solver; ///< The saddle point solver.

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
        z  = zeros(n);
        sx = zeros(n);
        sy = zeros(m);
        sa = zeros(n);
        sb = zeros(m);

        // Initialize the indices of the rows in matrix A that only have positive or zero coefficients.
        std::vector<Index> iposrows;
        for(auto i = 0; i < ml; ++i)
            if(args.A.row(i).minCoeff() >= 0.0)
                iposrows.push_back(i);
        ipositiverows = Indices::Map(iposrows.data(), iposrows.size());

        // Initialize the indices of the rows in matrix A that only have negative or zero coefficients.
        std::vector<Index> inegrows;
        for(auto i = 0; i < ml; ++i)
            if(args.A.row(i).maxCoeff() <= 0.0)
                inegrows.push_back(i);
        inegativerows = Indices::Map(inegrows.data(), inegrows.size());
    }

    /// Initialize the step calculator before calling decompose multiple times.
    auto initialize(StepperInitializeArgs args) -> void
    {
        // Auxiliary const references
        const auto b      = args.b;
        const auto xlower = args.xlower;
        const auto xupper = args.xupper;
        const auto A      = W.topRows(ml);

        // Auxiliary references
        auto& stability = args.stability;
        auto x          = args.x;

        // Ensure consistent dimensions of vectors/matrices.
        assert(b.rows() == ml);
        assert(xlower.rows() == n);
        assert(xupper.rows() == n);
        assert(x.rows() == n);

        //======================================================================
        // IDENTIFY STRICTLY LOWER/UPPER UNSTABLE VARIABLES
        //======================================================================
        // These are variables that need to be strictly imposed on their bounds
        // because it is not possible to attain feasibility. This happens
        // linear equality constraints with strictly positive (or negative)
        // coefficients cannot be satisfied with primal values that are inside
        // the feasible domain.
        //======================================================================
        /// The flags for each variable indicating if they have been marked strictly lower/upper unstable.
        std::vector<bool> is_strictly_lower_unstable(n, false);
        std::vector<bool> is_strictly_upper_unstable(n, false);

        for(auto i : ipositiverows)
            if(A.row(i)*xlower >= b[i])
                for(auto j = 0; j < n; ++j)
                    if(A(i, j) != 0.0)
                        is_strictly_lower_unstable[j] = true;

        for(auto i : ipositiverows)
            if(A.row(i)*xupper <= b[i])
                for(auto j = 0; j < n; ++j)
                    if(A(i, j) != 0.0)
                        is_strictly_upper_unstable[j] = true;

        for(auto i : inegativerows)
            if(A.row(i)*xlower <= b[i])
                for(auto j = 0; j < n; ++j)
                    if(A(i, j) != 0.0)
                        is_strictly_lower_unstable[j] = true;

        for(auto i : inegativerows)
            if(A.row(i)*xupper >= b[i])
                for(auto j = 0; j < n; ++j)
                    if(A(i, j) != 0.0)
                        is_strictly_upper_unstable[j] = true;

        // Reset the order of the species
        Indices iordering(n);
        for(auto i = 0; i < n; ++i)
            iordering[i] = i;

        // Organize the variables so that strictly lower and upper unstable ones are in the back.
        auto is_strictly_lower_unstable_fn = [&](Index i) { return is_strictly_lower_unstable[i]; };
        auto is_strictly_upper_unstable_fn = [&](Index i) { return is_strictly_upper_unstable[i]; };

        const auto pos0 = n;
        const auto pos1 = moveRightIf(iordering.head(pos0), is_strictly_upper_unstable_fn);
        const auto pos2 = moveRightIf(iordering.head(pos1), is_strictly_lower_unstable_fn);

        // Compute the number of strictly upper and lower unstable variables
        const auto nsuu = pos0 - pos1;
        const auto nslu = pos1 - pos2;

        // The indices of the strictly lower and upper unstable variables
        const auto isuu = iordering.head(pos0).tail(nsuu);
        const auto islu = iordering.head(pos1).tail(nslu);

        stability.update({ iordering, n - nslu - nsuu, 0, 0, nslu, nsuu });

        // Attach the strictly unstable variables to either their upper or lower bounds
        x(isuu) = xupper(isuu);
        x(islu) = xlower(islu);
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
        auto& stability = args.stability;
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
        assert(W.rows() == m);
        assert(W.cols() == n || m == 0);

        // Update the coefficient matrix W = [A; J] with the updated J block
        W.bottomRows(mn) = J;

        // Calculate the optimality residuals
        z.noalias() = g + tr(W)*y;

        // The function that computes the z-threshold for variable xi to determine its stability.
        const auto zeps = [&](auto i)
        {
            // The threshold below is computed taking into account the order of
            // magnitude of g[i] as well as m = rows(W), because of
            // accumulation of round-off errors from tr(W)*y.
            const auto eps = std::numeric_limits<double>::epsilon();
            return (1 + abs(g[i])) * eps * m;
        };

        // Update the ordering of the variables with lower and upper bounds
        auto is_lower_unstable_fn = [&](Index i) { return x[i] == xlower[i] && z[i] > 0.0; };
        auto is_upper_unstable_fn = [&](Index i) { return x[i] == xupper[i] && z[i] < 0.0; };

        Indices iordering = stability.indicesVariables();

        const auto nslu = stability.numStrictlyLowerUnstableVariables();
        const auto nsuu = stability.numStrictlyUpperUnstableVariables();

        // Organize the primal variables in the order: (stable, lower unstable, upper unstable, strictly lower unstable, strictly upper unstable).
        const auto pos0 = n - nslu - nsuu;
        const auto pos1 = moveRightIf(iordering.head(pos0), is_upper_unstable_fn);
        const auto pos2 = moveRightIf(iordering.head(pos1), is_lower_unstable_fn);

        // Update the number of upper unstable, lower unstable, and stable variables
        const auto nuu  = pos0 - pos1;
        const auto nlu  = pos1 - pos2;
        const auto ns   = pos2;

        stability.update({ iordering, ns, nlu, nuu, nslu, nsuu });

        // The indices of all unstable variables. These will be classified as
        // fixed variables when solving the saddle point problem to compute the
        // Newton step. This effectively reduces the dimension of the linear
        // algebra problem solved (i.e. the unstable variables do not make up
        // to the final dimension of the matrix equations solved).
        const auto iu = stability.indicesUnstableVariables();

        // Decompose the saddle point matrix.
        // This decomposition is later used in method solve, possibly many
        // times! Consider lower/upper unstable variables as "fixed" variables
        // in the saddle point problem. Reason: we do not need to compute
        // Newton steps for the currently unstable variables!
        solver.decompose({ H, J, Matrix{}, iu });
    }

    /// Solve the saddle point problem.
    auto solve(StepperSolveArgs args) -> void
    {
        // Auxiliary constant references
        const auto x          = args.x;
        const auto y          = args.y;
        const auto b          = args.b;
        const auto g          = args.g;
        const auto H          = args.H;
        const auto h          = args.h;
        const auto A          = W.topRows(ml);
        const auto J          = W.bottomRows(mn);
        const auto& stability = args.stability;

        // Auxiliary references
        auto dx = args.dx;
        auto dy = args.dy;
        auto rx = args.rx;
        auto ry = args.ry;
        auto z  = args.z;

        // Ensure consistent dimensions of vectors/matrices.
        assert(x.rows() == n);
        assert(y.rows() == m);
        assert(g.rows() == n);
        assert(A.rows() == ml);
        assert(A.cols() == n || ml == 0);

        // The indices of all unstable variables
        auto iu = stability.indicesUnstableVariables();

        // The indices of the strictly lower and upper unstable variables
        auto isu = stability.indicesStrictlyUnstableVariables();

        // Calculate the instability measure of the variables.
        z.noalias() = g + tr(W)*y;

        // Calculate the residuals of the first-order optimality conditions
        rx = -z;

        // Set residuals with respect to unstable variables to zero. This
        // ensures that they are not taken into account when checking for
        // convergence.
        rx(iu).fill(0.0);

        // Calculate the residuals of the feasibility conditions
        ry.head(ml).noalias() = -(A*x - b);
        ry.tail(mn).noalias() = -h;

        // // For the strictly unstable variables, however, set the values in
        // // vector `a` to zero. This is to ensure that the strictly unstable
        // // variables are not even taken into account in the calculation, not
        // // even in the linear equality constraints. It is like if they were not
        // // part of the problem.
        // sa(isu).fill(0.0);

        // Solve the saddle point problem.
        // Note: For numerical accuracy, it is important to compute
        // directly the next x and y iterates, instead of dx and dy.
        // This is because the latter causes very small values on the
        // right-hand side of the saddle point problem, and algebraic
        // manipulation of these small values results in round-off errors.
        solver.solve({ H, J, x, g, b, h, sx, sy });

        // Finalize the computation of the steps dx and dy
        dx = sx - x;
        dy = sy - y;

        // Ensure the strictly unstable variables have zero steps, not even
        // round-off errors allowed.
        dx(isu).fill(0.0);

        //=====================================================================
        // Exponential Impulse with x' = x * exp(dx/x)
        //======================================================================
        // static bool firstiter = true;
        // const auto res = norm(A*dx)/norm(b);

        // if(res < eps && !firstiter)
        // if(res < options.tolerance)
        // {
        //     sx = x.array() * dx.cwiseQuotient(x).array().exp();
        //     dx = sx - x;
        // }
    }

    /// Compute the sensitivity derivatives of the saddle point problem.
    auto sensitivities(StepperSensitivitiesArgs args) -> void
    {
        // Auxiliary references
        auto dxdp       = args.dxdp;
        auto dydp       = args.dydp;
        auto dzdp       = args.dzdp;
        auto dgdp       = args.dgdp;
        auto dbdp       = args.dbdp;
        auto dhdp       = args.dhdp;
        auto& stability = args.stability;

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
        auto is = stability.indicesStableVariables();
        auto iu = stability.indicesUnstableVariables();
        // auto is = stability.iordering.head(stability.ns);
        // auto iu = stability.iordering.tail(n - stability.ns);

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
    pimpl->initialize(args);
}

auto Stepper::decompose(StepperDecomposeArgs args) -> void
{
    pimpl->decompose(args);
}

auto Stepper::solve(StepperSolveArgs args) -> void
{
    pimpl->solve(args);
}

auto Stepper::sensitivities(StepperSensitivitiesArgs args) -> void
{
    pimpl->sensitivities(args);
}

} // namespace Optima
