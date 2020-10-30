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
#include <Optima/MasterMatrix.hpp>
#include <Optima/MasterVector.hpp>

namespace Optima {

struct LinearSolver::Impl
{
    const Index nx = 0; ///< The number of variables x.
    const Index np = 0; ///< The number of variables p.
    const Index ny = 0; ///< The number of variables y.
    const Index nz = 0; ///< The number of variables z.
    const Index nw = 0; ///< The number of variables w = (y, z).

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

    Impl(Index nx, Index np, Index ny, Index nz)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz),
      rangespace(nx, np, ny, nz),
      nullspace(nx, np, ny, nz),
      fullspace(nx, np, ny, nz),
      x(zeros(nx)), p(zeros(np)), w(zeros(nw)), wbar(zeros(nw)),
      ax(zeros(nx)), aw(zeros(nw))
    {
    }

    /// Decompose the canonical matrix.
    auto decomposeCanonical(CanonicalMatrix Mc) -> void
    {
        switch(options.method)
        {
        case LinearSolverMethod::Nullspace: nullspace.decompose(Mc); break;
        case LinearSolverMethod::Rangespace: rangespace.decompose(Mc); break;
        default: fullspace.decompose(Mc); break;
        }
    }

    /// Solve the canonical linear problem.
    auto solveCanonical(CanonicalMatrix Mc, CanonicalVector ac, CanonicalVectorRef uc) -> void
    {
        switch(options.method)
        {
        case LinearSolverMethod::Nullspace: nullspace.solve(Mc, ac, uc); break;
        case LinearSolverMethod::Rangespace: rangespace.solve(Mc, ac, uc); break;
        default: fullspace.solve(Mc, ac, uc); break;
        }
    }

    auto decompose(const MasterMatrix& M) -> void
    {
        decomposeCanonical(M.canonicalMatrix());
    }

    auto solve(const MasterMatrix& M, const MasterVector& a, MasterVector& u) -> void
    {
        const auto Mbar = M.canonicalForm();

        const auto dims = Mbar.dims;
        const auto R    = Mbar.R;
        const auto js   = Mbar.js;
        const auto ju   = Mbar.ju;
        const auto Wu   = Mbar.Wu;

        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;

        auto as = ax.head(ns);
        auto au = ax.tail(nu);

        as.noalias() = a.x(js);
        au.noalias() = a.x(ju);

        auto awbs = aw.head(nbs);
        aw.noalias() = R * a.w;

        const auto ap = a.p;

        solve(M, {as, ap, awbs}, u);

        u.x(ju) = au;
    }

    auto solve(const MasterMatrix& M, CanonicalVector a, MasterVector& u) -> void
    {
        const auto Mc = M.canonicalMatrix();
        const auto Mbar = M.canonicalForm();

        const auto dims = Mbar.dims;
        const auto R    = Mbar.R;
        const auto js   = Mbar.js;
        const auto ju   = Mbar.ju;

        const auto ns  = dims.ns;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nl  = dims.nl;

        auto xs = x.head(ns);

        auto wbs = wbar.head(nbs);
        auto wbu = wbar.segment(nbs, nbu);
        auto wbl = wbar.tail(nl);

        const auto as = a.xs;
        const auto ap = a.p;
        const auto awbs = a.wbs.head(nbs);

        solveCanonical(Mc, {as, ap, awbs}, {xs, p, wbs});

        wbu.fill(0.0);
        wbl.fill(0.0);

        w.noalias() = tr(R) * wbar;

        u.x(js) = xs;
        u.x(ju).fill(0.0);
        u.p = p;
        u.y = w.head(ny);
        u.z = w.tail(nz);
    }
};

LinearSolver::LinearSolver(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz))
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

auto LinearSolver::decompose(const MasterMatrix& M) -> void
{
    pimpl->decompose(M);
}

auto LinearSolver::solve(const MasterMatrix& M, const MasterVector& a, MasterVector& u) -> void
{
    pimpl->solve(M, a, u);
}

auto LinearSolver::solve(const MasterMatrix& M, CanonicalVector a, MasterVector& u) -> void
{
    pimpl->solve(M, a, u);
}

} // namespace Optima
