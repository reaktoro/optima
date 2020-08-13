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

// Eigen includes
#include <Optima/deps/eigen3/Eigen/src/LU/PartialPivLU.h>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverFullspace::Impl
{
    Matrix mat; ///< The matrix used as a workspace for the decompose and solve methods.
    Vector vec; ///< The vector used as a workspace for the decompose and solve methods

    Eigen::PartialPivLU<Matrix> lu; ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverFullspace::Impl instance.
    Impl(Index nx, Index np, Index m)
    : mat(nx + np + m, nx + np + m), vec(nx + np + m)
    {}

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        // The dimension variables needed below
        const auto ns  = args.dims.ns;
        const auto nbs = args.dims.nbs;
        const auto nns = args.dims.nns;
        const auto np  = args.dims.np;

        // The matrix where the canonical saddle point matrix is assembled
        auto M = mat.topLeftCorner(ns + np + nbs, ns + np + nbs);

        // The 1st, 2nd, 3rd, 4th blocks of rows in M
        auto M1 = M.topRows(nbs);
        auto M2 = M.middleRows(nbs, nns);
        auto M3 = M.middleRows(nbs + nns, np);
        auto M4 = M.bottomRows(nbs);

        // The matrices Hbsbs, Hbsns, Hnsbs, Hnsns in Hss
        const auto Hbsbs = args.Hss.topRows(nbs).leftCols(nbs);
        const auto Hbsns = args.Hss.topRows(nbs).rightCols(nns);
        const auto Hnsbs = args.Hss.bottomRows(nns).leftCols(nbs);
        const auto Hnsns = args.Hss.bottomRows(nns).rightCols(nns);

        // The matrices Hbsnp and Hnsnp in in Hsp
        const auto Hbsnp = args.Hsp.topRows(nbs);
        const auto Hnsnp = args.Hsp.bottomRows(nns);

        // The matrices Sbsns and Sbsnp
        const auto Sbsns = args.Sbsns;
        const auto Sbsnp = args.Sbsnp;

        // The matrices Vnpbs, Vnpns, Vnpnp
        const auto Vnpbs = args.Hps.leftCols(nbs);
        const auto Vnpns = args.Hps.rightCols(nns);
        const auto Vnpnp = args.Hpp;

        // The identity matrix Ibsbs
        auto Ibsbs = identity(nbs, nbs);

        // The zero matrices 0npbs and 0bsbs
        auto Onpbs = zeros(np, nbs);
        auto Obsbs = zeros(nbs, nbs);

        // Assemble the matrix M
        if(nbs) M1 << Hbsbs, Hbsns, Hbsnp, Ibsbs;
        if(nns) M2 << Hnsbs, Hnsns, Hnsnp, tr(Sbsns);
        if( np) M3 << Vnpbs, Vnpns, Vnpnp, Onpbs;
        if(nbs) M4 << Ibsbs, Sbsns, Sbsnp, Obsbs;

        // Compute the LU decomposition of M
        lu.compute(M);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        // The dimension variables needed below
        const auto ns  = args.dims.ns;
        const auto nbs = args.dims.nbs;
        const auto nns = args.dims.nns;
        const auto np  = args.dims.np;

        // The vector where the right-hand side of the linear system is assembled
        auto r = vec.head(ns + np + nbs);

        // Assemble the right-hand side vector r = (as, ap, bbs)
        r << args.as, args.ap, args.bbs;

        // Solve the system of linear equations using the LU decomposition of M.
        r.noalias() = lu.solve(r);

        // Get the result of xs, p, ybs from r, which now holds r = (xs, p, ybs)
        args.xs  = r.head(ns);
        args.p   = r.segment(ns, np);
        args.ybs = r.tail(nbs);
    }
};

SaddlePointSolverFullspace::SaddlePointSolverFullspace(Index nx, Index np, Index m)
: pimpl(new Impl(nx, np, m))
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
