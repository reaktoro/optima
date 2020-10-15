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

#include "SaddlePointSolverFullspace.hpp"

// C++ includes
#include <cassert>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/LU.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverFullspace::Impl
{
    Matrix mat; ///< The matrix used as a workspace for the decompose and solve methods.
    Vector vec; ///< The vector used as a workspace for the decompose and solve methods
    LU lu;      ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverFullspace::Impl instance.
    Impl(Index nx, Index np, Index nw)
    : mat(nx + np + nw, nx + np + nw), vec(nx + np + nw)
    {}

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        const auto ns  = args.dims.ns;
        const auto nbs = args.dims.nbs;
        const auto nns = args.dims.nns;
        const auto np  = args.dims.np;
        const auto nw  = args.dims.nw;

        const auto t = ns + np + nbs;

        auto M = mat.topLeftCorner(t, t);

        auto M1 = M.topRows(nbs);
        auto M2 = M.middleRows(nbs, nns);
        auto M3 = M.middleRows(nbs + nns, np);
        auto M4 = M.bottomRows(nbs);

        const auto Hbsbs = args.Hss.topRows(nbs).leftCols(nbs);
        const auto Hbsns = args.Hss.topRows(nbs).rightCols(nns);
        const auto Hnsbs = args.Hss.bottomRows(nns).leftCols(nbs);
        const auto Hnsns = args.Hss.bottomRows(nns).rightCols(nns);

        const auto Hbsp = args.Hsp.topRows(nbs);
        const auto Hnsp = args.Hsp.bottomRows(nns);

        const auto Sbsns = args.Sbsns;
        const auto Sbsp  = args.Sbsp;

        const auto Vpbs = args.Vps.leftCols(nbs);
        const auto Vpns = args.Vps.rightCols(nns);
        const auto Vpp = args.Vpp;

        const auto Ibsbs = identity(nbs, nbs);

        const auto Opbs  = zeros(np, nbs);
        const auto Obsbs = zeros(nbs, nbs);

        if(nbs) M1 << Hbsbs, Hbsns, Hbsp, Ibsbs;
        if(nns) M2 << Hnsbs, Hnsns, Hnsp, tr(Sbsns);
        if( np) M3 << Vpbs, Vpns, Vpp, Opbs;
        if(nbs) M4 << Ibsbs, Sbsns, Sbsp, Obsbs;

        lu.decompose(M);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        const auto ns  = args.dims.ns;
        const auto nbs = args.dims.nbs;
        const auto nns = args.dims.nns;
        const auto np  = args.dims.np;
        const auto nw  = args.dims.nw;

        const auto t = ns + np + nbs;

        auto r = vec.head(t);

        auto xbs = r.head(nbs);
        auto xns = r.segment(nbs, nns);
        auto p   = r.segment(nbs + nns, np);
        auto wbs = r.tail(nbs);

        r << args.as, args.ap, args.awbs;

        lu.solve(r);

        args.xs << xbs, xns;
        args.p = p;
        args.wbs = wbs;
    }
};

SaddlePointSolverFullspace::SaddlePointSolverFullspace(Index nx, Index np, Index nw)
: pimpl(new Impl(nx, np, nw))
{}

SaddlePointSolverFullspace::SaddlePointSolverFullspace(const SaddlePointSolverFullspace& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolverFullspace::~SaddlePointSolverFullspace()
{}

auto SaddlePointSolverFullspace::operator=(SaddlePointSolverFullspace other) -> SaddlePointSolverFullspace&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolverFullspace::decompose(CanonicalSaddlePointMatrix args) -> void
{
    return pimpl->decompose(args);
}

auto SaddlePointSolverFullspace::solve(CanonicalSaddlePointProblem args) -> void
{
    return pimpl->solve(args);
}

} // namespace Optima
