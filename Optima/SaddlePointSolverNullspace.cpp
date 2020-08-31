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
#include <Optima/deps/eigen3/Eigen/src/LU/FullPivLU.h>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverNullspace::Impl
{
    Vector ax;  ///< The workspace for the right-hand side vectors ax
    Vector ap;  ///< The workspace for the right-hand side vectors ap
    Vector az;  ///< The workspace for the right-hand side vectors az
    Vector b;   ///< The workspace for the right-hand side vectors b
    Matrix Hxx; ///< The workspace for the auxiliary matrices Hss.
    Matrix Hxp; ///< The workspace for the auxiliary matrices Hsp.
    Matrix Vpx; ///< The workspace for the auxiliary matrices Vps.
    Matrix Vpp; ///< The workspace for the auxiliary matrices Vpp.
    Matrix Jx;  ///< The workspace for the auxiliary matrices Js.
    Matrix Jp;  ///< The workspace for the auxiliary matrices Jp.
    Matrix Mw;  ///< The workspace for the matrix M in the decompose and solve methods.
    Vector rw;  ///< The workspace for the vector r in the decompose and solve methods.

    //======================================================================
    // Note: The full pivoting strategy is needed at the moment to resolve
    // singular matrices. Using a partial pivoting scheme via PartialPivLU
    // would need to be combined with a search for linearly dependent rows in
    // the produced upper triangular matrix U.
    //======================================================================

    Eigen::FullPivLU<Matrix> lu; ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverNullspace::Impl instance.
    Impl(Index nx, Index np, Index ny, Index nz)
    {
        // Allocate auxiliary vectors/matrices
        ax.resize(nx);
        ap.resize(np);
        az.resize(nz);
        b.resize(ny);
        Hxx.resize(nx, nx);
        Hxp.resize(nx, np);
        Vpx.resize(np, nx);
        Vpp.resize(np, np);
        Jx.resize(nz, nx);
        Jp.resize(nz, np);
        Mw.resize(nx + np + ny + nz, nx + np + ny + nz);
        rw.resize(nx + np + ny + nz);
    }

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        const auto ns  = args.dims.ns;
        const auto nbs = args.dims.nbs;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nns = args.dims.nns;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;
        const auto np  = args.dims.np;
        const auto nz  = args.dims.nz;

        auto Hss = Hxx.topLeftCorner(ns, ns);
        auto Hsp = Hxp.topRows(ns);
        auto Vps = Vpx.leftCols(ns);

        Hss = args.Hss;
        Hsp = args.Hsp;
        Vps = args.Vps;
        Vpp = args.Vpp;

        auto Hbsbs = Hss.topRows(nbs).leftCols(nbs);
        auto Hbsns = Hss.topRows(nbs).rightCols(nns);
        auto Hnsbs = Hss.bottomRows(nns).leftCols(nbs);
        auto Hnsns = Hss.bottomRows(nns).rightCols(nns);

        auto Hbebe = Hbsbs.topRows(nbe).leftCols(nbe);
        auto Hbebi = Hbsbs.topRows(nbe).rightCols(nbi);
        auto Hbibe = Hbsbs.bottomRows(nbi).leftCols(nbe);
        auto Hbibi = Hbsbs.bottomRows(nbi).rightCols(nbi);

        auto Hbens = Hbsns.topRows(nbe);
        auto Hbins = Hbsns.bottomRows(nbi);

        auto Hnsbe = Hnsbs.leftCols(nbe);
        auto Hnsbi = Hnsbs.rightCols(nbi);

        auto Hbsp = Hsp.topRows(nbs);
        auto Hnsp = Hsp.bottomRows(nns);

        auto Hbep = Hbsp.topRows(nbe);
        auto Hbip = Hbsp.bottomRows(nbi);

        auto Vpbs = Vps.leftCols(nbs);
        auto Vpns = Vps.rightCols(nns);

        auto Vpbe = Vpbs.leftCols(nbe);
        auto Vpbi = Vpbs.rightCols(nbi);

        auto Js = Jx.leftCols(ns);

        Js = args.Js;
        Jp = args.Jp;

        auto Jbs = Js.leftCols(nbs);
        auto Jns = Js.rightCols(nns);
        auto Jbe = Jbs.leftCols(nbe);
        auto Jne = Jns.leftCols(nne);
        auto Jbi = Jbs.rightCols(nbi);
        auto Jni = Jns.rightCols(nni);

        const auto Sbsns = args.Sbsns;
        const auto Sbsp  = args.Sbsp;

        const auto Sbens = Sbsns.topRows(nbe);
        const auto Sbins = Sbsns.bottomRows(nbi);

        const auto Sbep = Sbsp.topRows(nbe);
        const auto Sbip = Sbsp.bottomRows(nbi);

        const auto Ibebe = identity(nbe, nbe);
        const auto Opz   = zeros(np, nz);
        const auto Opbe  = zeros(np, nbe);
        const auto Obebe = zeros(nbe, nbe);
        const auto Obez  = zeros(nbe, nz);
        const auto Ozbe  = zeros(nz, nbe);
        const auto Ozz   = zeros(nz, nz);

        const auto t = nbe + nns + np + nz + nbe;

        auto M = Mw.topLeftCorner(t, t);

        auto M1 = M.topRows(nbe);
        auto M2 = M.middleRows(nbe, nns);
        auto M3 = M.middleRows(nbe + nns, np);
        auto M4 = M.middleRows(nbe + nns + np, nz);
        auto M5 = M.bottomRows(nbe);

        Hbins.noalias() -= Hbibi * Sbins;
        Hbens.noalias() -= Hbebi * Sbins;
        Hnsns.noalias() -= Hnsbi * Sbins;
        Vpns.noalias()  -= Vpbi * Sbins;
        Jns.noalias()   -= Jbi * Sbins;

        Hbip.noalias() -= Hbibi * Sbip;
        Hbep.noalias() -= Hbebi * Sbip;
        Hnsp.noalias() -= Hnsbi * Sbip;
        Vpp.noalias()  -= Vpbi * Sbip;
        Jp.noalias()   -= Jbi * Sbip;

        Hnsbe.noalias() -= tr(Sbins) * Hbibe;
        Hnsns.noalias() -= tr(Sbins) * Hbins;
        Hnsp.noalias()  -= tr(Sbins) * Hbip;

        if(nbe) M1 << Hbebe, Hbens, Hbep, tr(Jbe), Ibebe;
        if(nns) M2 << Hnsbe, Hnsns, Hnsp, tr(Jns), tr(Sbens);
        if( np) M3 << Vpbe, Vpns, Vpp, Opz, Opbe;
        if( nz) M4 << Jbe, Jns, Jp, Ozz, Ozbe;
        if(nbe) M5 << Ibebe, Sbens, Sbep, Obez, Obebe;

        lu.compute(M);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        const auto ns  = args.dims.ns;
        const auto nbs = args.dims.nbs;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nns = args.dims.nns;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;
        const auto np  = args.dims.np;
        const auto nz  = args.dims.nz;

        const auto Hss = Hxx.topLeftCorner(ns, ns);
        const auto Hsp = Hxp.topRows(ns);
        const auto Vps = Vpx.leftCols(ns);

        const auto Hbsbs = Hss.topRows(nbs).leftCols(nbs);
        const auto Hbsns = Hss.topRows(nbs).rightCols(nns);
        const auto Hnsbs = Hss.bottomRows(nns).leftCols(nbs);

        const auto Hbebi = Hbsbs.topRows(nbe).rightCols(nbi);
        const auto Hbibe = Hbsbs.bottomRows(nbi).leftCols(nbe);
        const auto Hbibi = Hbsbs.bottomRows(nbi).rightCols(nbi);

        const auto Hbins = Hbsns.bottomRows(nbi);
        const auto Hnsbi = Hnsbs.rightCols(nbi);

        const auto Hbsp = Hsp.topRows(nbs);
        const auto Hbip = Hbsp.bottomRows(nbi);

        const auto Vpbs = Vps.leftCols(nbs);
        const auto Vpbi = Vpbs.rightCols(nbi);

        const auto Js = Jx.leftCols(ns);
        const auto Jbs = Js.leftCols(nbs);
        const auto Jbi = Jbs.rightCols(nbi);

        const auto Sbsns = args.Sbsns;
        const auto Sbsp  = args.Sbsp;

        const auto Sbens = Sbsns.topRows(nbe);
        const auto Sbins = Sbsns.bottomRows(nbi);

        const auto Sbep = Sbsp.topRows(nbe);
        const auto Sbip = Sbsp.bottomRows(nbi);

        auto as  = ax.head(ns);
        auto abs = as.head(nbs);
        auto ans = as.tail(nns);
        auto abe = abs.head(nbe);
        auto abi = abs.tail(nbi);

        auto bbs = b.head(nbs);
        auto bbe = bbs.head(nbe);
        auto bbi = bbs.tail(nbi);

        as = args.as;
        ap = args.ap;
        az = args.az;
        bbs = args.aybs;

        abi.noalias() -= Hbibi * bbi;
        abe.noalias() -= Hbebi * bbi;
        ans.noalias() -= Hnsbi * bbi;
        ap.noalias()  -= Vpbi * bbi;
        az.noalias()  -= Jbi * bbi;

        ans -= tr(Sbins) * abi;

        const auto t = nbe + nns + np + nz + nbe;

        auto r = rw.head(t);

        auto xbe = r.head(nbe);
        auto xns = r.segment(nbe, nns);
        auto p   = r.segment(nbe + nns, np);
        auto z   = r.segment(nbe + nns + np, nz);
        auto ybe = r.tail(nbe);

        r << abe, ans, ap, az, bbe;

        r.noalias() = lu.solve(r);

        auto xbi = bbi;
        auto ybi = abi;

        xbi.noalias() = bbi - Sbins*xns - Sbip*p;
        ybi.noalias() = abi - Hbibe*xbe - Hbins*xns - Hbip*p - tr(Jbi)*z;

        args.xs << xbe, xbi, xns;
        args.ybs << ybe, ybi;
        args.p = p;
        args.z = z;
    }
};

SaddlePointSolverNullspace::SaddlePointSolverNullspace(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz))
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
