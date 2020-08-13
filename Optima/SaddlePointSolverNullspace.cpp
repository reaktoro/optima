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

#include "SaddlePointSolverNullspace.hpp"

// C++ includes
#include <cassert>

// Eigen includes
#include <Optima/deps/eigen3/Eigen/src/LU/PartialPivLU.h>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverNullspace::Impl
{
    Vector ax;  ///< The workspace for the right-hand side vectors ax
    Vector ap;  ///< The workspace for the right-hand side vectors ap
    Vector b;  ///< The workspace for the right-hand side vectors b
    Matrix Hxx; ///< The workspace for the auxiliary matrices Hss.
    Matrix Hxp; ///< The workspace for the auxiliary matrices Hsp.
    Matrix Vpx; ///< The workspace for the auxiliary matrices Vps.
    Matrix Vpp; ///< The workspace for the auxiliary matrices Vpp.
    Matrix Mw;  ///< The workspace for the matrix M in the decompose and solve methods.
    Vector rw;  ///< The workspace for the vector r in the decompose and solve methods.

    Eigen::PartialPivLU<Matrix> lu; ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverNullspace::Impl instance.
    Impl(Index nx, Index np, Index m)
    {
        // Allocate auxiliary vectors/matrices
        ax.resize(nx);
        ap.resize(np);
        b.resize(m);
        Hxx.resize(nx, nx);
        Hxp.resize(nx, np);
        Vpx.resize(np, nx);
        Vpp.resize(np, np);
        Mw.resize(nx + np + m, nx + np + m);
        rw.resize(nx + np + m);
    }

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        // The dimension variables needed below
        const auto ns  = args.dims.ns;
        const auto nbs = args.dims.nbs;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nns = args.dims.nns;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;
        const auto np  = args.dims.np;

        // Views to auxiliary workspace matrices Hss, Hsp, Vps
        auto Hss = Hxx.topLeftCorner(ns, ns);
        auto Hsp = Hxp.topRows(ns);
        auto Vps = Vpx.leftCols(ns);

        // Initialize the auxiliary workspace matrices Hss, Hsp, Vps, Vpp
        Hss = args.Hss;
        Hsp = args.Hsp;
        Vps = args.Hps;
        Vpp = args.Hpp;

        // The matrix blocks in Hss = [Hbsbs Hbsns; Hnsbs Hnsns]
        auto Hbsbs = Hss.topRows(nbs).leftCols(nbs);
        auto Hbsns = Hss.topRows(nbs).rightCols(nns);
        auto Hnsbs = Hss.bottomRows(nns).leftCols(nbs);
        auto Hnsns = Hss.bottomRows(nns).rightCols(nns);

        // The matrix blocks in Hbsbs = [Hbebe Hbebi; Hbibe Hbibi]
        auto Hbebe = Hbsbs.topRows(nbe).leftCols(nbe);
        auto Hbebi = Hbsbs.topRows(nbe).rightCols(nbi);
        auto Hbibe = Hbsbs.bottomRows(nbi).leftCols(nbe);
        auto Hbibi = Hbsbs.bottomRows(nbi).rightCols(nbi);

        // The matrix blocks in Hbsns = [Hbens; Hbins]
        auto Hbens = Hbsns.topRows(nbe);
        auto Hbins = Hbsns.bottomRows(nbi);

        // The matrix blocks in Hnsbs = [Hnsbe Hnsbi]
        auto Hnsbe = Hnsbs.leftCols(nbe);
        auto Hnsbi = Hnsbs.rightCols(nbi);

        // The matrix blocks in Hsp = [Hbsnp; Hnsnp]
        auto Hbsnp = Hsp.topRows(nbs);
        auto Hnsnp = Hsp.bottomRows(nns);

        // The matrix blocks in Hbsnp = [Hbenp; Hbinp]
        auto Hbenp = Hbsnp.topRows(nbe);
        auto Hbinp = Hbsnp.bottomRows(nbi);

        // The matrices Vnpbs, Vnpns
        auto Vnpbs = Vps.leftCols(nbs);
        auto Vnpns = Vps.rightCols(nns);

        // The matrix blocks in Vnpbs = [Vnpbe Vnpbi]
        auto Vnpbe = Vnpbs.leftCols(nbe);
        auto Vnpbi = Vnpbs.rightCols(nbi);

        // The matrix Vnpnp = Vpp
        auto Vnpnp = Vpp.topLeftCorner(np, np);

        // The matrices Sbsns and Sbsnp
        const auto Sbsns = args.Sbsns;
        const auto Sbsnp = args.Sbsnp;

        // The matrix blocks in Sbsns = [Sbens; Sbins]
        const auto Sbens = Sbsns.topRows(nbe);
        const auto Sbins = Sbsns.bottomRows(nbi);

        // The matrix blocks in Sbsnp = [Sbenp; Sbinp]
        const auto Sbenp = Sbsnp.topRows(nbe);
        const auto Sbinp = Sbsnp.bottomRows(nbi);

        // The auxiliary matrices Ibebe, Onpbe, Obebe
        const auto Ibebe = identity(nbe, nbe);
        const auto Onpbe = zeros(np, nbe);
        const auto Obebe = zeros(nbe, nbe);

        // The number of variables in the linear system
        const auto t = nbe + nns + np + nbe;

        // The coefficient matrix in the linear system
        auto M = Mw.topLeftCorner(t, t);

        // Views to the rowwise blocks of M
        auto M1 = M.topRows(nbe);
        auto M2 = M.middleRows(nbe, nns);
        auto M4 = M.middleRows(nbe + nns, np);
        auto M5 = M.bottomRows(nbe);

        // Perform the sequence of alterations in H and V matrices
        Hbins.noalias() -= Hbibi * Sbins;
        Hbens.noalias() -= Hbebi * Sbins;
        Hnsns.noalias() -= Hnsbi * Sbins;
        Vnpns.noalias() -= Vnpbi * Sbins;

        Hbinp.noalias() -= Hbibi * Sbinp;
        Hbenp.noalias() -= Hbebi * Sbinp;
        Hnsnp.noalias() -= Hnsbi * Sbinp;
        Vnpnp.noalias() -= Vnpbi * Sbinp;

        Hnsbe -= tr(Sbins) * Hbibe;
        Hnsns -= tr(Sbins) * Hbins;
        Hnsnp -= tr(Sbins) * Hbinp;

        // Assemble the matrix M
        if(nbe) M1 << Hbebe, Hbens, Hbenp, Ibebe;
        if(nns) M2 << Hnsbe, Hnsns, Hnsnp, tr(Sbens);
        if( np) M4 << Vnpbe, Vnpns, Vnpnp, Onpbe;
        if(nbe) M5 << Ibebe, Sbens, Sbenp, Obebe;

        // Compute the LU decomposition of M.
        lu.compute(M);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        // The dimension variables needed below
        const auto ns  = args.dims.ns;
        const auto nbs = args.dims.nbs;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nns = args.dims.nns;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;
        const auto np  = args.dims.np;

        // Views to auxiliary workspace matrices Hss, Hsp, Vps
        const auto Hss = Hxx.topLeftCorner(ns, ns);
        const auto Hsp = Hxp.topRows(ns);
        const auto Vps = Vpx.leftCols(ns);

        // The matrix blocks in Hss = [Hbsbs Hbsns; Hnsbs Hnsns]
        const auto Hbsbs = Hss.topRows(nbs).leftCols(nbs);
        const auto Hbsns = Hss.topRows(nbs).rightCols(nns);
        const auto Hnsbs = Hss.bottomRows(nns).leftCols(nbs);
        const auto Hnsns = Hss.bottomRows(nns).rightCols(nns);

        // The matrix blocks in Hbsbs = [Hbebe Hbebi; Hbibe Hbibi]
        const auto Hbebe = Hbsbs.topRows(nbe).leftCols(nbe);
        const auto Hbebi = Hbsbs.topRows(nbe).rightCols(nbi);
        const auto Hbibe = Hbsbs.bottomRows(nbi).leftCols(nbe);
        const auto Hbibi = Hbsbs.bottomRows(nbi).rightCols(nbi);

        // The matrix blocks in Hbsns = [Hbens; Hbins]
        const auto Hbens = Hbsns.topRows(nbe);
        const auto Hbins = Hbsns.bottomRows(nbi);

        // The matrix blocks in Hnsbs = [Hnsbe Hnsbi]
        const auto Hnsbe = Hnsbs.leftCols(nbe);
        const auto Hnsbi = Hnsbs.rightCols(nbi);

        // The matrix blocks in Hsp = [Hbsnp; Hnsnp]
        const auto Hbsnp = Hsp.topRows(nbs);
        const auto Hnsnp = Hsp.bottomRows(nns);

        // The matrix blocks in Hbsnp = [Hbenp; Hbinp]
        const auto Hbenp = Hbsnp.topRows(nbe);
        const auto Hbinp = Hbsnp.bottomRows(nbi);

        // The matrices Vnpbs, Vnpns
        const auto Vnpbs = Vps.leftCols(nbs);
        const auto Vnpns = Vps.rightCols(nns);

        // The matrix blocks in Vnpbs = [Vnpbe Vnpbi]
        const auto Vnpbe = Vnpbs.leftCols(nbe);
        const auto Vnpbi = Vnpbs.rightCols(nbi);

        // The matrix Vnpnp = Vpp
        const auto Vnpnp = Vpp.topLeftCorner(np, np);

        // The matrices Sbsns and Sbsnp
        const auto Sbsns = args.Sbsns;
        const auto Sbsnp = args.Sbsnp;

        // The matrix blocks in Sbsns = [Sbens; Sbins]
        const auto Sbens = Sbsns.topRows(nbe);
        const auto Sbins = Sbsns.bottomRows(nbi);

        // The matrix blocks in Sbsnp = [Sbenp; Sbinp]
        const auto Sbenp = Sbsnp.topRows(nbe);
        const auto Sbinp = Sbsnp.bottomRows(nbi);

        // The auxiliary vector bbs
        auto bbs = b.head(nbs);

        // The vector blocks in bbs = (bbe, bbi)
        auto bbe = bbs.head(nbe);
        auto bbi = bbs.tail(nbi);

        // The auxiliary vector as
        auto as = ax.head(ns);

        // The vector blocks in as = (abs, ans)
        auto abs = as.head(nbs);
        auto ans = as.tail(nns);

        // The vector blocks in abs = (abe, abi)
        auto abe = abs.head(nbe);
        auto abi = abs.tail(nbi);

        // Initialize vectors as = (abs, ans), ap, bbs
        as = args.as;
        ap = args.ap;
        bbs = args.bbs;

        // Compute the first round of modifications in as and ap
        abi -= Hbibi*bbi;
        abe -= Hbebi*bbi;
        ans -= Hnsbi*bbi;
        ap  -= Vnpbi*bbi;

        // Compute the second round of modifications in ans
        ans -= tr(Sbins)*abi;

        // The number of variables in the linear system
        const auto t = nbe + nns + np + nbe;

        // The right-hand side vector in the linear system
        auto r = rw.head(t);

        // The vector blocks in r = (xbe, xns, xnp, ybe)
        auto xbe = r.head(nbe);
        auto xns = r.segment(nbe, nns);
        auto xnp = r.segment(nbe + nns, np);
        auto ybe = r.tail(nbe);

        // Assemble the right-hand side vector r = (abe, ans, anp, bbe)
        r << abe, ans, ap, bbe;

        // Solve the system of linear equations
        r.noalias() = lu.solve(r);

        // Use bbi and abi as workspace for computing xbi and ybi
        auto xbi = bbi;
        auto ybi = abi;

        // Compute the remaining variables xbi and ybi using the other ones
        xbi.noalias() = bbi - Sbins*xns - Sbinp*xnp;
        ybi.noalias() = abi - Hbibe*xbe - Hbins*xns - Hbinp*xnp;

        // Finalize the computation of xs = (xbs, xns), p, and ybs
        args.xs << xbe, xbi, xns;
        args.ybs << ybe, ybi;
        args.p = xnp;
    }
};

SaddlePointSolverNullspace::SaddlePointSolverNullspace(Index nx, Index np, Index m)
: pimpl(new Impl(nx, np, m))
{}

SaddlePointSolverNullspace::SaddlePointSolverNullspace(const SaddlePointSolverNullspace& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolverNullspace::~SaddlePointSolverNullspace()
{}

auto SaddlePointSolverNullspace::operator=(SaddlePointSolverNullspace other) -> SaddlePointSolverNullspace&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolverNullspace::decompose(CanonicalSaddlePointMatrix args) -> void
{
    return pimpl->decompose(args);
}

auto SaddlePointSolverNullspace::solve(CanonicalSaddlePointProblem args) -> void
{
    return pimpl->solve(args);
}

} // namespace Optima
