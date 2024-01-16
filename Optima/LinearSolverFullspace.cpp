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
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "LinearSolverFullspace.hpp"

// C++ includes
#include <cassert>

// Optima includes
#include <Optima/CanonicalVector.hpp>
#include <Optima/CanonicalMatrix.hpp>
#include <Optima/Exception.hpp>
#include <Optima/LU.hpp>

namespace Optima {

struct LinearSolverFullspace::Impl
{
    Matrix mat; ///< The matrix used as a workspace for the decompose and solve methods.
    Vector vec; ///< The vector used as a workspace for the decompose and solve methods
    LU lu;      ///< The LU decomposition solver.

    Impl()
    {}

    auto decompose(CanonicalMatrix J) -> void
    {
        const auto dims = J.dims;

        const auto ns  = dims.ns;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;
        const auto np  = dims.np;
        const auto nw  = dims.nw;
        const auto nt  = dims.nt;

        const auto t = ns + np + nbs;

        mat.resize(nt, nt);
        auto M = mat.topLeftCorner(t, t);

        auto M1 = M.topRows(nbs);
        auto M2 = M.middleRows(nbs, nns);
        auto M3 = M.middleRows(nbs + nns, np);
        auto M4 = M.bottomRows(nbs);

        const auto Hbsbs = J.Hss.topRows(nbs).leftCols(nbs);
        const auto Hbsns = J.Hss.topRows(nbs).rightCols(nns);
        const auto Hnsbs = J.Hss.bottomRows(nns).leftCols(nbs);
        const auto Hnsns = J.Hss.bottomRows(nns).rightCols(nns);

        const auto Hbsp = J.Hsp.topRows(nbs);
        const auto Hnsp = J.Hsp.bottomRows(nns);

        const auto Sbsns = J.Sbsns;
        const auto Sbsp  = J.Sbsp;

        const auto Vpbs = J.Vps.leftCols(nbs);
        const auto Vpns = J.Vps.rightCols(nns);
        const auto Vpp = J.Vpp;

        const auto Ibsbs = identity(nbs, nbs);

        const auto Opbs  = zeros(np, nbs);
        const auto Obsbs = zeros(nbs, nbs);

        if(nbs) M1 << Hbsbs, Hbsns, Hbsp, Ibsbs;
        if(nns) M2 << Hnsbs, Hnsns, Hnsp, tr(Sbsns);
        if( np) M3 << Vpbs, Vpns, Vpp, Opbs;
        if(nbs) M4 << Ibsbs, Sbsns, Sbsp, Obsbs;

        lu.decompose(M);
    }

    auto solve(CanonicalMatrix J, CanonicalVectorView a, CanonicalVectorRef u) -> void
    {
        const auto dims = J.dims;

        const auto ns  = dims.ns;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;
        const auto np  = dims.np;
        const auto nw  = dims.nw;
        const auto nt  = dims.nt;

        const auto t = ns + np + nbs;

        vec.resize(nt);
        auto r = vec.head(t);

        auto xbs = r.head(nbs);
        auto xns = r.segment(nbs, nns);
        auto p   = r.segment(nbs + nns, np);
        auto wbs = r.tail(nbs);

        const auto axs  = a.xs;
        const auto ap   = a.p;
        const auto awbs = a.wbs;

        r << axs, ap, awbs;

        lu.solve(r);

        u.xs << xbs, xns;
        u.p = p;
        u.wbs = wbs;
    }
};

LinearSolverFullspace::LinearSolverFullspace()
: pimpl(new Impl())
{}

LinearSolverFullspace::LinearSolverFullspace(const LinearSolverFullspace& other)
: pimpl(new Impl(*other.pimpl))
{}

LinearSolverFullspace::~LinearSolverFullspace()
{}

auto LinearSolverFullspace::operator=(LinearSolverFullspace other) -> LinearSolverFullspace&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto LinearSolverFullspace::decompose(CanonicalMatrix M) -> void
{
    pimpl->decompose(M);
}

auto LinearSolverFullspace::solve(CanonicalMatrix J, CanonicalVectorView a, CanonicalVectorRef u) -> void
{
    pimpl->solve(J, a, u);
}

} // namespace Optima
