// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more Mbar.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "LinearSolver.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/LinearSolverFullspace.hpp>
#include <Optima/LinearSolverNullspace.hpp>
#include <Optima/LinearSolverRangespace.hpp>

namespace Optima {

struct LinearSolver::Impl
{
    LinearSolverOptions options; ///< The options for the linear solver.

    LinearSolverRangespace rangespace; ///< The linear solver based on a rangespace algorithm.
    LinearSolverNullspace nullspace;   ///< The linear solver based on a nullspace algorithm.
    LinearSolverFullspace fullspace;   ///< The linear solver based on a fullspace algorithm.

    Vector x; ///< The auxiliary solution vector x.
    Vector p; ///< The auxiliary solution vector p.
    Vector w; ///< The auxiliary solution vector w.

    Vector wbar; ///< The auxiliary solution vector w' = (w'bs, w'bu, w'bl).

    Vector ax; ///< The auxiliary solution vector ax.
    Vector aw; ///< The auxiliary solution vector aw.

    Impl()
    {}

    auto solveCanonical(CanonicalMatrix Mc, CanonicalVectorView ac, CanonicalVectorRef uc) -> void
    {
        switch(options.method)
        {
        case LinearSolverMethod::Nullspace: nullspace.solve(Mc, ac, uc); break;
        case LinearSolverMethod::Rangespace: rangespace.solve(Mc, ac, uc); break;
        default: fullspace.solve(Mc, ac, uc); break;
        }
    }

    auto decompose(CanonicalMatrix Mc) -> void
    {
        switch(options.method)
        {
        case LinearSolverMethod::Nullspace: nullspace.decompose(Mc); break;
        case LinearSolverMethod::Rangespace: rangespace.decompose(Mc); break;
        default: fullspace.decompose(Mc); break;
        }
    }

    auto solve(CanonicalMatrix Mc, MasterVectorView a, MasterVectorRef u) -> void
    {
        const auto dims = Mc.dims;
        const auto Rbs  = Mc.Rbs;
        const auto js   = Mc.js;
        const auto ju   = Mc.ju;

        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;

        ax.resize(dims.nx);
        auto as = ax.head(ns);
        auto au = ax.tail(nu);

        as.noalias() = a.x(js);
        au.noalias() = a.x(ju);

        aw.resize(dims.nw);
        auto awbs = aw.head(nbs);
        awbs = Rbs * a.w;

        const auto ap = a.p;

        solve(Mc, {as, au, ap, awbs}, u);
    }

    auto solve(CanonicalMatrix Mc, CanonicalVectorView a, MasterVectorRef u) -> void
    {
        const auto dims = Mc.dims;
        const auto Rbs  = Mc.Rbs;
        const auto js   = Mc.js;
        const auto ju   = Mc.ju;

        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto nbs = dims.nbs;
        const auto nl  = dims.nl;

        p.resize(dims.np);
        x.resize(dims.nx);
        auto xs = x.head(ns);
        auto xu = x.tail(nu);

        wbar.resize(dims.nw);
        auto wbs = wbar.head(nbs);

        const auto as = a.xs;
        const auto au = a.xu;
        const auto ap = a.p;
        const auto awbs = a.wbs.head(nbs);

        solveCanonical(Mc, {as, au, ap, awbs}, {xs, xu, p, wbs});

        w.noalias() = tr(Rbs) * wbs;

        u.x(js) = xs;
        u.x(ju) = au;
        u.p = p;
        u.w = w;
    }
};

LinearSolver::LinearSolver()
: pimpl(new Impl())
{}

LinearSolver::LinearSolver(const LinearSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

LinearSolver::~LinearSolver()
{}

auto LinearSolver::operator=(LinearSolver other) -> LinearSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto LinearSolver::setOptions(const LinearSolverOptions& options) -> void
{
    pimpl->options = options;
}

auto LinearSolver::options() const -> const LinearSolverOptions&
{
    return pimpl->options;
}

auto LinearSolver::decompose(CanonicalMatrix Mc) -> void
{
    pimpl->decompose(Mc);
}

auto LinearSolver::solve(CanonicalMatrix Mc, MasterVectorView a, MasterVectorRef u) -> void
{
    pimpl->solve(Mc, a, u);
}

auto LinearSolver::solve(CanonicalMatrix Mc, CanonicalVectorView ac, MasterVectorRef u) -> void
{
    pimpl->solve(Mc, ac, u);
}

} // namespace Optima
