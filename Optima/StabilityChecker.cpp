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

#include "StabilityChecker.hpp"

// C++ includes
#include <vector>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>

namespace Optima {

using std::abs;
using std::vector;

struct StabilityChecker::Impl
{
    Matrix W;                ///< The coefficient matrix W = [A; J] of the linear/nonlinear equality constraints.
    Vector z;                ///< The instability measures of the variables defined as z = g + tr(W)*y.
    Index n  = 0;            ///< The number of variables in x.
    Index ml = 0;            ///< The number of linear equality constraints.
    Index mn = 0;            ///< The number of non-linear equality constraints.
    Index m  = 0;            ///< The number of equality constraints (m = ml + mn).
    Index t  = 0;            ///< The total number of variables in x and y (t = n + m).
    Indices ipositiverows;   ///< The indices of the rows in matrix A that only have positive or zero coefficients.
    Indices inegativerows;   ///< The indices of the rows in matrix A that only have negative or zero coefficients.
    Indices iordering;       ///< The ordering of the primal variables *x* as *stable*, *lower unstable*, *upper unstable*, *strictly lower unstable*, *strictly upper unstable*.
    vector<bool> bslu;       ///< The flags indicating strictly lower unstable status for each variable.
    vector<bool> bsuu;       ///< The flags indicating strictly upper unstable status for each variable.
    Stability stability;     ///< The stability state of the primal variables *x*.

    /// Construct a default StabilityChecker::Impl instance.
    Impl()
    {}

    /// Construct a StabilityChecker::Impl instance.
    Impl(StabilityCheckerInitArgs args)
    : n(args.n), m(args.m), W(args.m, args.n)
    {
        // Ensure the step calculator is initialized with a positive number of variables.
        Assert(n > 0, "Could not proceed with StabilityChecker initialization.",
            "The number of variables is zero.");

        // Initialize the number of linear and nonlinear equality constraints
        ml = args.A.rows();
        mn = m - ml;

        // Initialize the matrix W = [A; J], with J=0 at this initialization time (updated at each decompose call)
        W << args.A, zeros(mn, n);

        // Initialize total number of variables x and y
        t  = n + m;

        // Initialize auxiliary vectors
        z  = zeros(n);

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

    /// Initialize the stability checker.
    auto initialize(StabilityCheckerInitializeArgs args) -> void
    {
        // Auxiliary references
        const auto A      = args.A;
        const auto b      = args.b;
        const auto xlower = args.xlower;
        const auto xupper = args.xupper;

        // Ensure consistent dimensions of vectors/matrices.
        assert(A.cols() == n || ml == 0);
        assert(A.rows() == ml);
        assert(b.rows() == ml);
        assert(xlower.rows() == n);
        assert(xupper.rows() == n);

        //======================================================================
        // IDENTIFY STRICTLY LOWER/UPPER UNSTABLE VARIABLES
        //======================================================================
        // These are variables that need to be strictly imposed on their bounds
        // because it is not possible to attain feasibility. This happens
        // linear equality constraints with strictly positive (or negative)
        // coefficients cannot be satisfied with primal values that are inside
        // the feasible domain.
        //======================================================================
        bslu.resize(n, false);
        bsuu.resize(n, false);

        for(auto i : ipositiverows)
            if(A.row(i)*xlower >= b[i])
                for(auto j = 0; j < n; ++j)
                    if(A(i, j) != 0.0)
                        bslu[j] = true;

        for(auto i : ipositiverows)
            if(A.row(i)*xupper <= b[i])
                for(auto j = 0; j < n; ++j)
                    if(A(i, j) != 0.0)
                        bsuu[j] = true;

        for(auto i : inegativerows)
            if(A.row(i)*xlower <= b[i])
                for(auto j = 0; j < n; ++j)
                    if(A(i, j) != 0.0)
                        bslu[j] = true;

        for(auto i : inegativerows)
            if(A.row(i)*xupper >= b[i])
                for(auto j = 0; j < n; ++j)
                    if(A(i, j) != 0.0)
                        bsuu[j] = true;

        // Reset the order of the variables
        iordering.resize(n);
        for(auto i = 0; i < n; ++i)
            iordering[i] = i;

        // Organize the variables so that strictly lower and upper unstable ones are in the back.
        auto is_strictly_lower_unstable_fn = [&](Index i) { return bslu[i]; };
        auto is_strictly_upper_unstable_fn = [&](Index i) { return bsuu[i]; };

        const auto pos0 = n;
        const auto pos1 = moveRightIf(iordering.head(pos0), is_strictly_upper_unstable_fn);
        const auto pos2 = moveRightIf(iordering.head(pos1), is_strictly_lower_unstable_fn);

        // Compute the number of strictly upper and lower unstable variables
        const auto nsuu = pos0 - pos1;
        const auto nslu = pos1 - pos2;

        // The indices of the strictly lower and upper unstable variables
        const auto isuu = iordering.head(pos0).tail(nsuu);
        const auto islu = iordering.head(pos1).tail(nslu);

        // Update the stability state of the variables
        stability.update({ iordering, n - nslu - nsuu, 0, 0, nslu, nsuu });
    }

    /// Update the stability checker.
    auto update(StabilityCheckerUpdateArgs args) -> void
    {
        // Auxiliary references
        auto W          = args.W;
        auto x          = args.x;
        auto y          = args.y;
        auto g          = args.g;
        auto xlower     = args.xlower;
        auto xupper     = args.xupper;
        const auto A    = W.topRows(ml);

        // Ensure consistent dimensions of vectors/matrices.
        assert(W.cols() == n || m == 0);
        assert(W.rows() == m);
        assert(x.rows() == n);
        assert(y.rows() == m);
        assert(g.rows() == n);
        assert(xlower.rows() == n);
        assert(xupper.rows() == n);

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
        auto is_lower_unstable_fn = [&](Index i) { return x[i] == xlower[i] && z[i] > -zeps(i); };
        auto is_upper_unstable_fn = [&](Index i) { return x[i] == xupper[i] && z[i] < +zeps(i); };

        iordering = stability.indicesVariables();

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

        // Update the stability state of the variables
        stability.update({ iordering, ns, nlu, nuu, nslu, nsuu });
    }
};

StabilityChecker::StabilityChecker()
: pimpl(new Impl())
{}

StabilityChecker::StabilityChecker(StabilityCheckerInitArgs args)
: pimpl(new Impl(args))
{}

StabilityChecker::StabilityChecker(const StabilityChecker& other)
: pimpl(new Impl(*other.pimpl))
{}

StabilityChecker::~StabilityChecker()
{}

auto StabilityChecker::operator=(StabilityChecker other) -> StabilityChecker&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto StabilityChecker::initialize(StabilityCheckerInitializeArgs args) -> void
{
    pimpl->initialize(args);
}

auto StabilityChecker::update(StabilityCheckerUpdateArgs args) -> void
{
    pimpl->update(args);
}

auto StabilityChecker::stability() const -> const Stability&
{
    return pimpl->stability;
}

} // namespace Optima
