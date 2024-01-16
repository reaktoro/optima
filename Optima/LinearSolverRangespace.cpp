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

#include "LinearSolverRangespace.hpp"

// C++ includes
#include <cassert>

// Optima includes
#include <Optima/CanonicalVector.hpp>
#include <Optima/CanonicalMatrix.hpp>
#include <Optima/Exception.hpp>
#include <Optima/LU.hpp>

namespace Optima {

struct LinearSolverRangespace::Impl
{
    Vector ax;        ///< The workspace for the right-hand side vector ax
    Vector ap;        ///< The workspace for the right-hand side vector ap
    Vector aw;        ///< The workspace for the right-hand side vector aw
    Vector Hd;        ///< The workspace for the diagonal entries in the Hss matrix.
    Matrix Tw;        ///< The workspace for the Tbb = Sbn*inv(Hnn)*tr(Sbn) matrix.
    Matrix Mw;        ///< The workspace for the M matrix in decompose and solve methods.
    Vector rw;        ///< The workspace for the r vector in solve method.
    Vector sw;        ///< The workspace for the s vector in solve method.
    Matrix barHsp;    ///< The workspace for matrix bar(Hsp)
    Matrix barVps;    ///< The workspace for matrix bar(Vps)
    Matrix barSbsns;  ///< The workspace for matrix bar(Sbsns)
    LU lu;            ///< The LU decomposition solver.

    Impl()
    {}

    auto decompose(CanonicalMatrix J) -> void
    {
        const auto dims = J.dims;

        const auto nx  = dims.nx;
        const auto ns  = dims.ns;
        const auto np  = dims.np;
        const auto nw  = dims.nw;
        const auto nt  = dims.nt;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;
        const auto nbe = dims.nbe;
        const auto nbi = dims.nbi;
        const auto nne = dims.nne;
        const auto nni = dims.nni;

        const auto Sbsns = J.Sbsns;
        const auto Sbene = Sbsns.topLeftCorner(nbe, nne);
        const auto Sbeni = Sbsns.topRightCorner(nbe, nni);
        const auto Sbine = Sbsns.bottomLeftCorner(nbi, nne);
        const auto Sbini = Sbsns.bottomRightCorner(nbi, nni);
        const auto Sbsne = Sbsns.leftCols(nne);

        const auto Sbsp = J.Sbsp;
        const auto Sbep = Sbsp.topRows(nbe);
        const auto Sbip = Sbsp.bottomRows(nbi);

        const auto Hbsp = J.Hsp.topRows(nbs);
        const auto Hnsp = J.Hsp.bottomRows(nns);
        const auto Hbep = Hbsp.topRows(nbe);
        const auto Hnep = Hnsp.topRows(nne);
        const auto Hbip = Hbsp.bottomRows(nbi);
        const auto Hnip = Hnsp.bottomRows(nni);

        const auto Vpp  = J.Vpp;
        const auto Vpbs = J.Vps.leftCols(nbs);
        const auto Vpns = J.Vps.rightCols(nns);
        const auto Vpbe = Vpbs.leftCols(nbe);
        const auto Vpne = Vpns.leftCols(nne);
        const auto Vpbi = Vpbs.rightCols(nbi);
        const auto Vpni = Vpns.rightCols(nni);

        const auto Ibebe = identity(nbe, nbe);
        const auto Ibibi = identity(nbi, nbi);

        Hd.resize(nx);
        auto Hs = Hd.head(ns);

        Hs = J.Hss.diagonal();

        const auto Hbsbs = Hs.head(nbs);
        const auto Hnsns = Hs.tail(nns);
        const auto Hbebe = Hbsbs.head(nbe);
        const auto Hbibi = Hbsbs.tail(nbi);
        const auto Hnene = Hnsns.head(nne);
        const auto Hnini = Hnsns.tail(nni);

        const auto invHbebe = diag(inv(Hbebe));
        const auto invHnene = diag(inv(Hnene));

        barHsp.resize(nx, np);
        barVps.resize(np, nx);
        barSbsns.resize(nw, nx);

        auto barHbep  = barHsp.topRows(nbe);
        auto barHnep  = barHsp.bottomRows(nne);
        auto barVpbe  = barVps.leftCols(nbe);
        auto barVpne  = barVps.rightCols(nne);
        auto barSbsne = barSbsns.topLeftCorner(nbs, nne);
        auto barSbene = barSbsne.topRows(nbe);
        auto barSbine = barSbsne.bottomRows(nbi);

        barHbep  = invHbebe * Hbep;
        barHnep  = invHnene * Hnep;
        barVpbe  = Vpbe * invHbebe;
        barVpne  = Vpne * invHnene;
        barSbene = Sbene * invHnene;
        barSbine = Sbine * invHnene;

        Tw.resize(nw, nw);
        auto Tbsbs = Tw.topLeftCorner(nbs, nbs);

        Tbsbs.noalias() = Sbsne * tr(barSbsne);

        const auto Tbibi = Tbsbs.bottomRightCorner(nbi, nbi);
        const auto Tbibe = Tbsbs.bottomLeftCorner(nbi, nbe);
        const auto Tbebi = Tbsbs.topRightCorner(nbe, nbi);
        const auto Tbebe = Tbsbs.topLeftCorner(nbe, nbe);

        const auto t = np + nbi + nbe + nni;

        Mw.resize(nt, nt);
        auto M = Mw.topLeftCorner(t, t);

        //======================================================================
        // IMPORTANT NOTE
        //======================================================================
        // The organization of the matrix blocks below have been carefully
        // determined. Before changing it, ensure this is really needed as the
        // accuracy of the solution of the linear system (and also its
        // robustness) can be compromised.
        //======================================================================

        auto M1 = M.topRows(np);
        auto M2 = M.middleRows(np, nbi);
        auto M3 = M.middleRows(np + nbi, nbe);
        auto M4 = M.bottomRows(nni);

        auto M11 = M1.leftCols(np);
        auto M12 = M1.middleCols(np, nbi);
        auto M13 = M1.middleCols(np + nbi, nbe);
        auto M14 = M1.rightCols(nni);

        auto M21 = M2.leftCols(np);
        auto M22 = M2.middleCols(np, nbi);
        auto M23 = M2.middleCols(np + nbi, nbe);
        auto M24 = M2.rightCols(nni);

        auto M31 = M3.leftCols(np);
        auto M32 = M3.middleCols(np, nbi);
        auto M33 = M3.middleCols(np + nbi, nbe);
        auto M34 = M3.rightCols(nni);

        auto M41 = M4.leftCols(np);
        auto M42 = M4.middleCols(np, nbi);
        auto M43 = M4.middleCols(np + nbi, nbe);
        auto M44 = M4.rightCols(nni);

        M11.noalias() = Vpp - Vpbe*barHbep - Vpne*barHnep + barVpne*tr(Sbine)*Hbip;
        M12.noalias() = Vpbi + barVpne*tr(Sbine)*diag(Hbibi);
        M13.noalias() = -barVpbe - barVpne*tr(Sbene);
        M14.noalias() = Vpni;

        M21.noalias() = Sbip - barSbine*Hnep + Tbibi*Hbip;
        M22.noalias() = Ibibi + Tbibi*diag(Hbibi);
        M23.noalias() = -Tbibe;
        M24.noalias() = Sbini;

        M31.noalias() = Sbep - barHbep - barSbene*Hnep + Tbebi*Hbip;
        M32.noalias() = Tbebi*diag(Hbibi);
        M33 = diag(inv(Hbebe)); M33 += Tbebe;
        M34.noalias() = Sbeni;

        M41.noalias() = Hnip - tr(Sbini)*Hbip;
        M42.noalias() = -tr(Sbini)*diag(Hbibi);
        M43.noalias() = tr(Sbeni);
        M44 = diag(Hnini);

        lu.decompose(M);
    }

    auto solve(CanonicalMatrix J, CanonicalVectorView a, CanonicalVectorRef u) -> void
    {
        const auto dims = J.dims;

        const auto nx  = dims.nx;
        const auto ns  = dims.ns;
        const auto np  = dims.np;
        const auto nw  = dims.nw;
        const auto nt  = dims.nt;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;
        const auto nbe = dims.nbe;
        const auto nbi = dims.nbi;
        const auto nne = dims.nne;
        const auto nni = dims.nni;

        const auto Sbsns = J.Sbsns;
        const auto Sbene = Sbsns.topLeftCorner(nbe, nne);
        const auto Sbeni = Sbsns.topRightCorner(nbe, nni);
        const auto Sbine = Sbsns.bottomLeftCorner(nbi, nne);
        const auto Sbini = Sbsns.bottomRightCorner(nbi, nni);
        const auto Sbsne = Sbsns.leftCols(nne);

        const auto Sbsp = J.Sbsp;
        const auto Sbep = Sbsp.topRows(nbe);
        const auto Sbip = Sbsp.bottomRows(nbi);

        const auto Hbsp = J.Hsp.topRows(nbs);
        const auto Hnsp = J.Hsp.bottomRows(nns);
        const auto Hbep = Hbsp.topRows(nbe);
        const auto Hnep = Hnsp.topRows(nne);
        const auto Hbip = Hbsp.bottomRows(nbi);
        const auto Hnip = Hnsp.bottomRows(nni);

        const auto Vpp  = J.Vpp;
        const auto Vpbs = J.Vps.leftCols(nbs);
        const auto Vpns = J.Vps.rightCols(nns);
        const auto Vpbe = Vpbs.leftCols(nbe);
        const auto Vpne = Vpns.leftCols(nne);
        const auto Vpbi = Vpbs.rightCols(nbi);
        const auto Vpni = Vpns.rightCols(nni);

        const auto Hs = Hd.head(ns);

        const auto Hbsbs = Hs.head(nbs);
        const auto Hnsns = Hs.tail(nns);
        const auto Hbebe = Hbsbs.head(nbe);
        const auto Hbibi = Hbsbs.tail(nbi);
        const auto Hnene = Hnsns.head(nne);
        const auto Hnini = Hnsns.tail(nni);

        const auto barVpne = barVps.rightCols(nne);

        const auto Tbsbs = Tw.topLeftCorner(nbs, nbs);
        const auto Tbibi = Tbsbs.bottomRightCorner(nbi, nbi);
        const auto Tbibe = Tbsbs.bottomLeftCorner(nbi, nbe);
        const auto Tbebi = Tbsbs.topRightCorner(nbe, nbi);
        const auto Tbebe = Tbsbs.topLeftCorner(nbe, nbe);

        ax.resize(nx);
        auto as  = ax.head(ns);
        auto abs = as.head(nbs);
        auto ans = as.tail(nns);
        auto abe = abs.head(nbe);
        auto ane = ans.head(nne);
        auto abi = abs.tail(nbi);
        auto ani = ans.tail(nni);

        aw.resize(nw);
        auto awbs = aw.head(nbs);
        auto awbe = awbs.head(nbe);
        auto awbi = awbs.tail(nbi);

        as = a.xs;
        ap = a.p;
        awbs = a.wbs;

        abe.noalias() = abe/Hbebe;
        ane.noalias() = ane/Hnene;

        ap.noalias()   = ap - Vpbe*abe - Vpne*ane + barVpne*tr(Sbine)*abi;
        awbi.noalias() = awbi - Sbine*ane + Tbibi*abi;
        awbe.noalias() = awbe - abe - Sbene*ane + Tbebi*abi;
        ani.noalias()  = ani - tr(Sbini)*abi;

        const auto t = np + nbi + nbe + nni;

        rw.resize(nt);
        sw.resize(nt);

        auto r = rw.head(t);
        auto s = sw.head(t);

        auto p   = r.head(np);
        auto xbi = r.segment(np, nbi);
        auto wbe = r.segment(np + nbi, nbe);
        auto xni = r.tail(nni);

        r << ap, awbi, awbe, ani;

        lu.solve(r);

        auto wbi = awbi;
        auto xbe = abe;
        auto xne = ane;

        wbi.noalias() = abi - Hbip*p - diag(Hbibi)*xbi;
        xbe.noalias() = abe - (Hbep*p + wbe) / Hbebe;
        xne.noalias() = ane - (Hnep*p + tr(Sbene)*wbe + tr(Sbine)*wbi) / Hnene;

        u.xs << xbe, xbi, xne, xni;
        u.p = p;
        u.wbs << wbe, wbi;
    }
};

LinearSolverRangespace::LinearSolverRangespace()
: pimpl(new Impl())
{}

LinearSolverRangespace::LinearSolverRangespace(const LinearSolverRangespace& other)
: pimpl(new Impl(*other.pimpl))
{}

LinearSolverRangespace::~LinearSolverRangespace()
{}

auto LinearSolverRangespace::operator=(LinearSolverRangespace other) -> LinearSolverRangespace&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto LinearSolverRangespace::decompose(CanonicalMatrix M) -> void
{
    pimpl->decompose(M);
}

auto LinearSolverRangespace::solve(CanonicalMatrix J, CanonicalVectorView a, CanonicalVectorRef u) -> void
{
    pimpl->solve(J, a, u);
}

} // namespace Optima
