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
    Matrix A;                ///< The coefficient matrix A = [Ax Ap].
    Vector s;                ///< The stability measures of the variables defined as s = g + tr(Ax)*y + tr(hx)*z.
    Index nx  = 0;           ///< The number of primal variables *x*.
    Index np  = 0;           ///< The number of parameter variables *p*.
    Index ny  = 0;           ///< The number of Lagrange variables *y*.
    Index nz  = 0;           ///< The number of Lagrange variables *z*.
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
    : nx(args.nx), np(args.np), ny(args.ny), nz(args.nz)
    {
        // Ensure the step calculator is initialized with a positive number of variables.
        Assert(nx > 0, "Could not proceed with StabilityChecker initialization.",
            "The number of primal variables x is zero.");

        // Initialize matrix A = [Ax Ap]
        A.resize(ny, nx + np);
        A << args.Ax, args.Ap;

        // Initialize auxiliary vectors
        s  = zeros(nx);

        // Initialize the indices of the rows in matrix A that only have positive or zero coefficients.
        std::vector<Index> iposrows;
        for(auto i = 0; i < ny; ++i)
            if(A.row(i).minCoeff() >= 0.0)
                iposrows.push_back(i);
        ipositiverows = Indices::Map(iposrows.data(), iposrows.size());

        // Initialize the indices of the rows in matrix A that only have negative or zero coefficients.
        std::vector<Index> inegrows;
        for(auto i = 0; i < ny; ++i)
            if(A.row(i).maxCoeff() <= 0.0)
                inegrows.push_back(i);
        inegativerows = Indices::Map(inegrows.data(), inegrows.size());
    }

    /// Initialize the stability checker.
    auto initialize(StabilityCheckerInitializeArgs args) -> void
    {
        // Auxiliary references
        const auto b      = args.b;
        const auto xlower = args.xlower;
        const auto xupper = args.xupper;
        const auto plower = args.plower;
        const auto pupper = args.pupper;

        // Ensure consistent dimensions of vectors/matrices.
        assert(b.rows() == ny);
        assert(xlower.rows() == nx);
        assert(xupper.rows() == nx);
        assert(plower.rows() == np);
        assert(pupper.rows() == np);

        // View to sub-matrices Ax and Ap in A = [Ax Ap]
        const auto Ax = A.leftCols(nx);
        const auto Ap = A.rightCols(np);

        //======================================================================
        // IDENTIFY STRICTLY LOWER/UPPER UNSTABLE VARIABLES
        //======================================================================
        // These are variables that need to be strictly imposed on their bounds
        // because it is not possible to attain feasibility. This happens
        // linear equality constraints with strictly positive (or negative)
        // coefficients cannot be satisfied with primal values that are inside
        // the feasible domain.
        //======================================================================
        bslu.resize(nx, false);
        bsuu.resize(nx, false);

        for(auto i : ipositiverows)
            if(Ax.row(i).dot(xlower) + Ap.row(i).dot(plower) >= b[i])
                for(auto j = 0; j < nx; ++j)
                    if(Ax(i, j) != 0.0)
                        bslu[j] = true;

        for(auto i : ipositiverows)
            if(Ax.row(i).dot(xupper) + Ap.row(i).dot(pupper) <= b[i])
                for(auto j = 0; j < nx; ++j)
                    if(Ax(i, j) != 0.0)
                        bsuu[j] = true;

        for(auto i : inegativerows)
            if(Ax.row(i).dot(xlower) + Ap.row(i).dot(plower) <= b[i])
                for(auto j = 0; j < nx; ++j)
                    if(Ax(i, j) != 0.0)
                        bslu[j] = true;

        for(auto i : inegativerows)
            if(Ax.row(i).dot(xupper) + Ap.row(i).dot(pupper) >= b[i])
                for(auto j = 0; j < nx; ++j)
                    if(Ax(i, j) != 0.0)
                        bsuu[j] = true;

        // Reset the order of the variables in x
        iordering.resize(nx);
        for(auto i = 0; i < nx; ++i)
            iordering[i] = i;

        // Organize the variables so that strictly lower and upper unstable ones are in the back.
        auto is_strictly_lower_unstable_fn = [&](Index i) { return bslu[i]; };
        auto is_strictly_upper_unstable_fn = [&](Index i) { return bsuu[i]; };

        const auto pos0 = nx;
        const auto pos1 = moveRightIf(iordering.head(pos0), is_strictly_upper_unstable_fn);
        const auto pos2 = moveRightIf(iordering.head(pos1), is_strictly_lower_unstable_fn);

        // Compute the number of strictly upper and lower unstable variables
        const auto nsuu = pos0 - pos1;
        const auto nslu = pos1 - pos2;

        // The indices of the strictly lower and upper unstable variables
        const auto isuu = iordering.head(pos0).tail(nsuu);
        const auto islu = iordering.head(pos1).tail(nslu);

        // Update the stability state of the variables
        stability.update({ iordering, nx - nslu - nsuu, 0, 0, nslu, nsuu });
    }

    /// Update the stability checker.
    auto update(StabilityCheckerUpdateArgs args) -> void
    {
        const auto x = args.x;
        const auto y = args.y;
        const auto z = args.z;
        const auto fx = args.fx;
        const auto hx = args.hx;
        const auto xlower = args.xlower;
        const auto xupper = args.xupper;

        assert(x.rows() == nx);
        assert(y.rows() == ny);
        assert(z.rows() == nz);
        assert(fx.rows() == nx);
        assert(hx.rows() == nz);
        assert(xlower.rows() == nx);
        assert(xupper.rows() == nx);

        const auto Ax = A.leftCols(nx);

        s.noalias() = fx + tr(Ax)*y + tr(hx)*z;

        // The function that computes the s-threshold for variable xi to determine its stability.
        const auto s_eps = [&](auto i)
        {
            // The threshold below is computed taking into account the order of
            // magnitude of fx[i] as well as m = ny + nz, because of
            // accumulation of round-off errors from tr(Ax)*y + tr(hx)*z.
            const auto eps = std::numeric_limits<double>::epsilon();
            const auto m = ny + nz;
            return (1 + abs(fx[i])) * eps * m;
        };

        auto is_lower_unstable_fn = [&](Index i) { return x[i] == xlower[i] && s[i] > -s_eps(i); };
        auto is_upper_unstable_fn = [&](Index i) { return x[i] == xupper[i] && s[i] < +s_eps(i); };

        iordering = stability.indicesVariables();

        const auto nslu = stability.numStrictlyLowerUnstableVariables();
        const auto nsuu = stability.numStrictlyUpperUnstableVariables();

        // Organize the primal variables in the order: (stable, lower unstable, upper unstable, strictly lower unstable, strictly upper unstable).
        const auto pos0 = nx - nslu - nsuu;
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
