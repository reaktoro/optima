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

#include "SaddlePointSolverRangespace.hpp"

// C++ includes
#include <cassert>



#include <iostream>
// #define PRINT(X) std::cout << #X " = \n" << X << "\n" << std::endl;
#define PRINT(X)




// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/LU.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverRangespace::Impl
{
    Vector ax;       ///< The workspace for the right-hand side vector ax
    Vector ap;       ///< The workspace for the right-hand side vector ap
    Vector az;       ///< The workspace for the right-hand side vector az
    Vector bb;       ///< The workspace for the right-hand side vectors bb
    Vector Hd;       ///< The workspace for the diagonal entries in the Hss matrix.
    Matrix Bw;       ///< The workspace for the Bnb = inv(Hnn)*tr(Sbn) matrix.
    Matrix Lw;       ///< The workspace for the Lbez = inv(Hbebe)*tr(Jbe) and Lnez = inv(Hnene)*tr(Jne) matrices.
    Matrix Tw;       ///< The workspace for the Tbb = Sbn*inv(Hnn)*tr(Sbn) matrix.
    Matrix Mw;       ///< The workspace for the M matrix in decompose and solve methods.
    Vector rw;       ///< The workspace for the r vector in solve method.
    Matrix barHsp;   ///< The workspace for matrix bar(Hsp)
    Matrix barVps;   ///< The workspace for matrix bar(Vps)
    Matrix barJs;    ///< The workspace for matrix bar(Js)
    Matrix barSbsns; ///< The workspace for matrix bar(Sbsns)
    LU lu;           ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverRangespace::Impl instance.
    Impl(Index nx, Index np, Index ny, Index nz)
    {
        ax.resize(nx);
        ap.resize(np);
        az.resize(nz);
        bb.resize(ny);
        Hd.resize(nx);
        Bw.resize(nx, ny);
        Lw.resize(nx, nz);
        Tw.resize(ny, ny);
        Mw.resize(nx + np + ny + nz, nx + np + ny + nz);
        rw.resize(nx + np + ny + nz);
        barHsp.resize(nx, np);
        barVps.resize(np, nx);
        barJs.resize(nz, nx);
        barSbsns.resize(ny, nx);
    }

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        const auto ns  = args.dims.ns;
        const auto np  = args.dims.np;
        const auto ny  = args.dims.ny;
        const auto nz  = args.dims.nz;
        const auto nbs = args.dims.nbs;
        const auto nns = args.dims.nns;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;

        const auto Sbsns = args.Sbsns;
        const auto Sbene = Sbsns.topLeftCorner(nbe, nne);
        const auto Sbeni = Sbsns.topRightCorner(nbe, nni);
        const auto Sbine = Sbsns.bottomLeftCorner(nbi, nne);
        const auto Sbini = Sbsns.bottomRightCorner(nbi, nni);
        const auto Sbsne = Sbsns.leftCols(nne);

        const auto Sbsp = args.Sbsp;
        const auto Sbep = Sbsp.topRows(nbe);
        const auto Sbip = Sbsp.bottomRows(nbi);

        const auto Hbsp = args.Hsp.topRows(nbs);
        const auto Hnsp = args.Hsp.bottomRows(nns);
        const auto Hbep = Hbsp.topRows(nbe);
        const auto Hnep = Hnsp.topRows(nne);
        const auto Hbip = Hbsp.bottomRows(nbi);
        const auto Hnip = Hnsp.bottomRows(nni);

        const auto Vpp  = args.Vpp;
        const auto Vpbs = args.Vps.leftCols(nbs);
        const auto Vpns = args.Vps.rightCols(nns);
        const auto Vpbe = Vpbs.leftCols(nbe);
        const auto Vpne = Vpns.leftCols(nne);
        const auto Vpbi = Vpbs.rightCols(nbi);
        const auto Vpni = Vpns.rightCols(nni);

        const auto Jp = args.Jp;
        const auto Js  = args.Js;
        const auto Jbs = Js.leftCols(nbs);
        const auto Jns = Js.rightCols(nns);
        const auto Jbe = Jbs.leftCols(nbe);
        const auto Jne = Jns.leftCols(nne);
        const auto Jbi = Jbs.rightCols(nbi);
        const auto Jni = Jns.rightCols(nni);

        const auto Ibebe = identity(nbe, nbe);
        const auto Ibibi = identity(nbi, nbi);

        auto Hs = Hd.head(ns);

        Hs = args.Hss.diagonal();

        const auto Hbsbs = Hs.head(nbs);
        const auto Hnsns = Hs.tail(nns);
        const auto Hbebe = Hbsbs.head(nbe);
        const auto Hbibi = Hbsbs.tail(nbi);
        const auto Hnene = Hnsns.head(nne);
        const auto Hnini = Hnsns.tail(nni);

        const auto invHbebe = diag(inv(Hbebe));
        const auto invHnene = diag(inv(Hnene));

        auto barHbep  = barHsp.topRows(nbe);
        auto barHnep  = barHsp.bottomRows(nne);
        auto barVpbe  = barVps.leftCols(nbe);
        auto barVpne  = barVps.rightCols(nne);
        auto barJbe   = barJs.leftCols(nbe);
        auto barJne   = barJs.rightCols(nne);
        auto barSbsne = barSbsns.topLeftCorner(nbs, nne);
        auto barSbene = barSbsne.topRows(nbe);
        auto barSbine = barSbsne.bottomRows(nbi);

        barHbep  = invHbebe * Hbep;
        barHnep  = invHnene * Hnep;
        barVpbe  = Vpbe * invHbebe;
        barVpne  = Vpne * invHnene;
        barJbe   = Jbe * invHbebe;
        barJne   = Jne * invHnene;
        barSbene = Sbene * invHnene;
        barSbine = Sbine * invHnene;

        auto Tbsbs = Tw.topLeftCorner(nbs, nbs);

        Tbsbs.noalias() = Sbsne * tr(barSbsne);

        const auto Tbibi = Tbsbs.bottomRightCorner(nbi, nbi);
        const auto Tbibe = Tbsbs.bottomLeftCorner(nbi, nbe);
        const auto Tbebi = Tbsbs.topRightCorner(nbe, nbi);
        const auto Tbebe = Tbsbs.topLeftCorner(nbe, nbe);

        const auto t = np + nz + nbi + nbe + nni;

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
        auto M2 = M.middleRows(np, nz);
        auto M3 = M.middleRows(np + nz, nbi);
        auto M4 = M.middleRows(np + nz + nbi, nbe);
        auto M5 = M.bottomRows(nni);

        auto M11 = M1.leftCols(np);
        auto M12 = M1.middleCols(np, nz);
        auto M13 = M1.middleCols(np + nz, nbi);
        auto M14 = M1.middleCols(np + nz + nbi, nbe);
        auto M15 = M1.rightCols(nni);

        auto M21 = M2.leftCols(np);
        auto M22 = M2.middleCols(np, nz);
        auto M23 = M2.middleCols(np + nz, nbi);
        auto M24 = M2.middleCols(np + nz + nbi, nbe);
        auto M25 = M2.rightCols(nni);

        auto M31 = M3.leftCols(np);
        auto M32 = M3.middleCols(np, nz);
        auto M33 = M3.middleCols(np + nz, nbi);
        auto M34 = M3.middleCols(np + nz + nbi, nbe);
        auto M35 = M3.rightCols(nni);

        auto M41 = M4.leftCols(np);
        auto M42 = M4.middleCols(np, nz);
        auto M43 = M4.middleCols(np + nz, nbi);
        auto M44 = M4.middleCols(np + nz + nbi, nbe);
        auto M45 = M4.rightCols(nni);

        auto M51 = M5.leftCols(np);
        auto M52 = M5.middleCols(np, nz);
        auto M53 = M5.middleCols(np + nz, nbi);
        auto M54 = M5.middleCols(np + nz + nbi, nbe);
        auto M55 = M5.rightCols(nni);

        M11.noalias() = Vpp - Vpbe*barHbep - Vpne*barHnep + barVpne*tr(Sbine)*Hbip;
        M12.noalias() = -barVpbe*tr(Jbe) - barVpne*tr(Jne) + barVpne*tr(Sbine)*tr(Jbi);
        M13.noalias() = Vpbi + barVpne*tr(Sbine)*diag(Hbibi);
        M14.noalias() = -barVpbe - barVpne*tr(Sbene);
        M15.noalias() = Vpni;

        M21.noalias() = Jp - Jbe*barHbep - Jne*barHnep + barJne*tr(Sbine)*Hbip;
        M22.noalias() = -barJbe*tr(Jbe) - barJne*tr(Jne) + barJne*tr(Sbine)*tr(Jbi);
        M23.noalias() = Jbi + barJne*tr(Sbine)*diag(Hbibi);
        M24.noalias() = -barJbe - barJne*tr(Sbene);
        M25.noalias() = Jni;

        M31.noalias() = Sbip - barSbine*Hnep + Tbibi*Hbip;
        M32.noalias() = -barSbine*tr(Jne) + Tbibi*tr(Jbi);
        M33.noalias() = Ibibi + Tbibi*diag(Hbibi);
        M34.noalias() = -Tbibe;
        M35.noalias() = Sbini;

        M41.noalias() = Sbep - barHbep - barSbene*Hnep + Tbebi*Hbip;
        M42.noalias() = -tr(barJbe) - barSbene*tr(Jne) + Tbebi*tr(Jbi);
        M43.noalias() = Tbebi*diag(Hbibi);
        M44 = diag(inv(-Hbebe)); M44 -= Tbebe;
        M45.noalias() = Sbeni;

        M51.noalias() = Hnip - tr(Sbini)*Hbip;
        M52.noalias() = tr(Jni) - tr(Sbini)*tr(Jbi);
        M53.noalias() = -tr(Sbini)*diag(Hbibi);
        M54.noalias() = tr(Sbeni);
        M55 = diag(Hnini);

        // M11.noalias() =  Vpp - Vpbe*invHbebe*Hbep - Vpne*invHnene*Hnep + Vpne*invHnene*tr(Sbine)*Hbip;
        // M12.noalias() = -Vpbe*invHbebe*tr(Jbe) - Vpne*invHnene*tr(Jne) + Vpne*invHnene*tr(Sbine)*tr(Jbi);
        // M13.noalias() =  Vpbi + Vpne*invHnene*tr(Sbine)*Hbibi;
        // M14.noalias() = -Vpbe*invHbebe - Vpne*invHnene*tr(Sbene);
        // M15.noalias() =  Vpni;

        // M21.noalias() = Jp - Jbe*invHbebe*Hbep - Jne*invHnene*Hnep + Jne*invHnene*tr(Sbine)*Hbip;
        // M22.noalias() = -Jbe*invHbebe*tr(Jbe) - Jne*invHnene*tr(Jne) + Jne*invHnene*tr(Sbine)*tr(Jbi);
        // M23.noalias() = Jbi + Jne*invHnene*tr(Sbine)*Hbibi;
        // M24.noalias() = -Jbe*invHbebe - Jne*invHnene*tr(Sbene);
        // M25.noalias() = Jni;

        // M31.noalias() = Sbip - Sbine*invHnene*Hnep + Sbine*invHnene*tr(Sbine)*Hbip;
        // M32.noalias() = -Sbine*invHnene*tr(Jne) + Sbine*invHnene*tr(Sbine)*tr(Jbi);
        // M33.noalias() = Ibibi + Sbine*invHnene*tr(Sbine)*Hbibi;
        // M34.noalias() = -Sbine*invHnene*tr(Sbene);
        // M35.noalias() = Sbini;

        // M41.noalias() = Sbep - invHbebe*Hbep - Sbene*invHnene*Hnep + Sbene*invHnene*tr(Sbine)*Hbip;
        // M42.noalias() = invHbebe*tr(-Jbe) - Sbene*invHnene*tr(Jne) + Sbene*invHnene*tr(Sbine)*tr(Jbi);
        // M43.noalias() = Sbene*invHnene*tr(Sbine)*Hbibi;
        // M44 = diag(inv(-Hbebe)); M44 -= Sbene*invHnene*tr(Sbene);
        // M45.noalias() = Sbeni;

        // M51.noalias() = Hnip - tr(Sbini)*Hbip;
        // M52.noalias() = tr(Jni) - tr(Sbini)*tr(Jbi);
        // M53.noalias() = -tr(Sbini)*Hbibi;
        // M54.noalias() = tr(Sbeni);
        // M55.noalias() = Hnini;






        // // Setting the 1st column of M with dimension nbe (corresponding to ybe)
        // Mbibi.noalias() = Ibibi + Tbibi * diag(Hbibi);
        // Mbebi.noalias() = Tbebi * diag(Hbibi);
        // Mnibi.noalias() = -tr(Sbini) * diag(Hbibi);
        // Mpbi.noalias() = Vpbi + Vpne * Bnebi * diag(Hbibi);

        // // Setting the 2nd column of M with dimension nbi (corresponding to xbi)
        // Mbibe.noalias() = -Tbibe;
        // Mbebe = diag(inv(-Hbebe)); Mbebe -= Tbebe;
        // Mnibe.noalias() = tr(Sbeni);
        // Mpbe.noalias() = -Vpbe * diag(inv(Hbebe)) - Vpne * Bnebe;

        // // Setting the 3rd column of M with dimension nni (corresponding to xni)
        // Mbini = Sbini;
        // Mbeni = Sbeni;
        // Mnini = diag(Hnini);
        // Mpni = Vpni;

        // // Setting the 4th column of M with dimension np (corresponding to p)
        // Mbip = Sbip;
        // Mbep = Sbep;
        // Mnip.fill(0.0);
        // Mpp = Vpp;

        // std::cout << "================================================================================================================================================" << std::endl;
        PRINT(M);
        // std::cout << "================================================================================================================================================" << std::endl;
        // PRINT(Mbebe);
        // PRINT(Mbibe);
        // PRINT(Mnibe);
        // PRINT(Mpbe);
        // PRINT(Mbebi);
        // PRINT(Mbibi);
        // PRINT(Mnibi);
        // PRINT(Mpbi);
        // PRINT(Mbeni);
        // PRINT(Mbini);
        // PRINT(Mnini);
        // PRINT(Mpni);
        // PRINT(Mbep);
        // PRINT(Mbip);
        // PRINT(Mnip);
        // PRINT(Mpp);

        lu.decompose(M);

        Matrix mlu = lu.matrixLU();

        const Matrix L = mlu.triangularView<Eigen::UnitLower>();
        const Matrix U = mlu.triangularView<Eigen::Upper>();

        PRINT(L);
        PRINT(U);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        const auto ns  = args.dims.ns;
        const auto np  = args.dims.np;
        const auto nz  = args.dims.nz;
        const auto nbs = args.dims.nbs;
        const auto nns = args.dims.nns;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;

        const auto Sbsns = args.Sbsns;
        const auto Sbene = Sbsns.topLeftCorner(nbe, nne);
        const auto Sbeni = Sbsns.topRightCorner(nbe, nni);
        const auto Sbine = Sbsns.bottomLeftCorner(nbi, nne);
        const auto Sbini = Sbsns.bottomRightCorner(nbi, nni);
        const auto Sbsne = Sbsns.leftCols(nne);

        const auto Sbsp = args.Sbsp;
        const auto Sbep = Sbsp.topRows(nbe);
        const auto Sbip = Sbsp.bottomRows(nbi);

        const auto Hbsp = args.Hsp.topRows(nbs);
        const auto Hnsp = args.Hsp.bottomRows(nns);
        const auto Hbep = Hbsp.topRows(nbe);
        const auto Hnep = Hnsp.topRows(nne);
        const auto Hbip = Hbsp.bottomRows(nbi);
        const auto Hnip = Hnsp.bottomRows(nni);

        const auto Vpp  = args.Vpp;
        const auto Vpbs = args.Vps.leftCols(nbs);
        const auto Vpns = args.Vps.rightCols(nns);
        const auto Vpbe = Vpbs.leftCols(nbe);
        const auto Vpne = Vpns.leftCols(nne);
        const auto Vpbi = Vpbs.rightCols(nbi);
        const auto Vpni = Vpns.rightCols(nni);

        const auto Jp = args.Jp;
        const auto Js  = args.Js;
        const auto Jbs = Js.leftCols(nbs);
        const auto Jns = Js.rightCols(nns);
        const auto Jbe = Jbs.leftCols(nbe);
        const auto Jne = Jns.leftCols(nne);
        const auto Jbi = Jbs.rightCols(nbi);
        const auto Jni = Jns.rightCols(nni);

        const auto Hs = Hd.head(ns);

        const auto Hbsbs = Hs.head(nbs);
        const auto Hnsns = Hs.tail(nns);
        const auto Hbebe = Hbsbs.head(nbe);
        const auto Hbibi = Hbsbs.tail(nbi);
        const auto Hnene = Hnsns.head(nne);
        const auto Hnini = Hnsns.tail(nni);

        const auto barVpne = barVps.rightCols(nne);
        const auto barJne  = barJs.rightCols(nne);

        const auto Tbsbs = Tw.topLeftCorner(nbs, nbs);
        const auto Tbibi = Tbsbs.bottomRightCorner(nbi, nbi);
        const auto Tbibe = Tbsbs.bottomLeftCorner(nbi, nbe);
        const auto Tbebi = Tbsbs.topRightCorner(nbe, nbi);
        const auto Tbebe = Tbsbs.topLeftCorner(nbe, nbe);

        auto as  = ax.head(ns);
        auto abs = as.head(nbs);
        auto ans = as.tail(nns);
        auto abe = abs.head(nbe);
        auto ane = ans.head(nne);
        auto abi = abs.tail(nbi);
        auto ani = ans.tail(nni);

        auto bbs = bb.head(nbs);
        auto bbe = bbs.head(nbe);
        auto bbi = bbs.tail(nbi);

        as = args.as;
        ap = args.ap;
        az = args.az;
        bbs = args.aybs;

        abe.noalias() = abe/Hbebe;
        ane.noalias() = ane/Hnene;

        ap.noalias()  = ap - Vpbe*abe - Vpne*ane + barVpne*tr(Sbine)*abi;
        az.noalias()  = az - Jbe*abe - Jne*ane + barJne*tr(Sbine)*abi;
        bbi.noalias() = bbi - Sbine*ane + Tbibi*abi;
        bbe.noalias() = bbe - abe - Sbene*ane + Tbebi*abi;
        ani.noalias() = ani - tr(Sbini)*abi;

        const auto t = np + nz + nbi + nbe + nni;

        auto r = rw.head(t);

        auto p   = r.head(np);
        auto z   = r.segment(np, nz);
        auto xbi = r.segment(np + nz, nbi);
        auto ybe = r.segment(np + nz + nbi, nbe);
        auto xni = r.tail(nni);

        r << ap, az, bbi, bbe, ani;

        PRINT(r);

        lu.solve(r);

        PRINT(r);

        const auto rank = lu.rank();

        assert(r.head(rank).allFinite()); // If FullPivLu is used, then this need to be checked based on the last indices of permutation matrix Q of the LU decomp PAQ = LU

        auto ybi = bbi;
        auto xbe = abe;
        auto xne = ane;

        ybi.noalias() = abi - Hbip*p - tr(Jbi)*z - diag(Hbibi)*xbi;
        xbe.noalias() = abe - (Hbep*p + tr(Jbe)*z + ybe) / Hbebe;
        xne.noalias() = ane - (Hnep*p + tr(Jne)*z + tr(Sbene)*ybe + tr(Sbine)*ybi) / Hnene;

        args.xs << xbe, xbi, xne, xni;
        args.p = p;
        args.z = z;
        args.ybs << ybe, ybi;
    }
};

SaddlePointSolverRangespace::SaddlePointSolverRangespace(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz))
{}

SaddlePointSolverRangespace::SaddlePointSolverRangespace(const SaddlePointSolverRangespace& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolverRangespace::~SaddlePointSolverRangespace()
{}

auto SaddlePointSolverRangespace::operator=(SaddlePointSolverRangespace other) -> SaddlePointSolverRangespace&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolverRangespace::decompose(CanonicalSaddlePointMatrix args) -> void
{
    return pimpl->decompose(args);
}

auto SaddlePointSolverRangespace::solve(CanonicalSaddlePointProblem args) -> void
{
    return pimpl->solve(args);
}

} // namespace Optima
