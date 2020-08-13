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

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/LU.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverRangespace::Impl
{
    Vector ax;  ///< The workspace for the right-hand side vector ax
    Vector ap;  ///< The workspace for the right-hand side vector ap
    Vector bb;  ///< The workspace for the right-hand side vectors bb
    Vector Hd;  ///< The workspace for the diagonal entries in the Hss matrix.
    Matrix Bw;  ///< The workspace for the Bnb matrix
    Matrix Tw;  ///< The workspace for the Tbb matrix
    Matrix Mw;  ///< The workspace for the M matrix in decompose and solve methods.
    Vector rw;  ///< The workspace for the r vector in solve method.
    LU lu;      ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverRangespace::Impl instance.
    Impl(Index nx, Index np, Index m)
    {
        // Allocate memory for auxiliary vectors/matrices
        ax.resize(nx);
        ap.resize(np);
        bb.resize(m);
        Hd.resize(nx);
        Bw.resize(nx, m);
        Tw.resize(m, m);
        Mw.resize(nx + np + m, nx + np + m);
        rw.resize(nx + np + m);
    }

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        // The dimension variables needed below
        const auto ns  = args.dims.ns;
        const auto np  = args.dims.np;
        const auto nbs = args.dims.nbs;
        const auto nns = args.dims.nns;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;

        // Views to the sub-matrices of Sbsns = [Sbene Sbeni; Sbine Sbini]
        const auto Sbsns = args.Sbsns;
        const auto Sbeni = Sbsns.topRightCorner(nbe, nni);
        const auto Sbini = Sbsns.bottomRightCorner(nbi, nni);
        const auto Sbsne = Sbsns.leftCols(nne);

        // Views to the sub-matrices of Sbsnp = [Sbenp Sbinp]
        const auto Sbsnp = args.Sbsnp;
        const auto Sbenp = Sbsnp.topRows(nbe);
        const auto Sbinp = Sbsnp.bottomRows(nbi);

        // View to the sub-matrix Vnpnp === Vpp
        const auto Vnpnp = args.Hpp;

        // View to the sub-matrices Vnpbs = [Vnpbe Vnpbi]
        const auto Vnpbs = args.Hps.leftCols(nbs);
        const auto Vnpbe = Vnpbs.leftCols(nbe);
        const auto Vnpbi = Vnpbs.rightCols(nbi);

        // View to the sub-matrices Vnpns = [Vnpne Vnpni]
        const auto Vnpns = args.Hps.rightCols(nns);
        const auto Vnpne = Vnpns.leftCols(nne);
        const auto Vnpni = Vnpns.rightCols(nni);

        // The identity matrices Ibebe and Ibibi
        const auto Ibebe = identity(nbe, nbe);
        const auto Ibibi = identity(nbi, nbi);

        // Views to the sub-vectors of Hs = [Hbebe Hbibi Hnene Hnini]
        auto Hs    = Hd.head(ns);
        auto Hbsbs = Hs.head(nbs);
        auto Hnsns = Hs.tail(nns);
        auto Hbebe = Hbsbs.head(nbe);
        auto Hbibi = Hbsbs.tail(nbi);
        auto Hnene = Hnsns.head(nne);
        auto Hnini = Hnsns.tail(nni);

        // Initialize Hs = [Hbebe Hbibi Hnene Hnini] above
        Hs = args.Hss.diagonal();

        // The auxiliary matrix Bnebs = inv(Hnene) * tr(Sbsne) and its submatrices
        auto Bnebs = Bw.topLeftCorner(nne, nbs);
        auto Bnebe = Bnebs.leftCols(nbe);
        auto Bnebi = Bnebs.rightCols(nbi);

        // Initialize Bnebs = inv(Hnene) * tr(Sbsne) above
        Bnebs.noalias() = diag(inv(Hnene)) * tr(Sbsne);

        // The auxiliary matrix Tbsbs = Sbsne * Bnebs and its submatrices
        auto Tbsbs = Tw.topLeftCorner(nbs, nbs);
        auto Tbibi = Tbsbs.bottomRightCorner(nbi, nbi);
        auto Tbibe = Tbsbs.bottomLeftCorner(nbi, nbe);
        auto Tbebi = Tbsbs.topRightCorner(nbe, nbi);
        auto Tbebe = Tbsbs.topLeftCorner(nbe, nbe);

        // Initialize Tbsbs = Sbsne * Bnebs above
        Tbsbs.noalias() = Sbsne * Bnebs;

        // The matrix M of the system of linear equations
        auto M = Mw.topLeftCorner(nbe + nbi + nni + np, nbe + nbi + nni + np);

        //======================================================================
        // IMPORTANT NOTE
        //======================================================================
        // The organization of the matrix blocks below have been carefully
        // determined. Before changing it, ensure this is really needed as the
        // accuracy of the solution of the linear system (and also its
        // robustness) can be compromised.
        //======================================================================

        auto Mbe = M.topRows(nbe);
        auto Mbi = M.middleRows(nbe, nbi);
        auto Mni = M.middleRows(nbe + nbi, nni);
        auto Mnp = M.bottomRows(np);

        auto Mbebe = Mbe.leftCols(nbe);
        auto Mbebi = Mbe.middleCols(nbe, nbi);
        auto Mbeni = Mbe.middleCols(nbe + nbi, nni);
        auto Mbenp = Mbe.rightCols(np);

        auto Mbibe = Mbi.leftCols(nbe);
        auto Mbibi = Mbi.middleCols(nbe, nbi);
        auto Mbini = Mbi.middleCols(nbe + nbi, nni);
        auto Mbinp = Mbi.rightCols(np);

        auto Mnibe = Mni.leftCols(nbe);
        auto Mnibi = Mni.middleCols(nbe, nbi);
        auto Mnini = Mni.middleCols(nbe + nbi, nni);
        auto Mninp = Mni.rightCols(np);

        auto Mnpbe = Mnp.leftCols(nbe);
        auto Mnpbi = Mnp.middleCols(nbe, nbi);
        auto Mnpni = Mnp.middleCols(nbe + nbi, nni);
        auto Mnpnp = Mnp.rightCols(np);

        //======================================================================
        // TODO: The current implementation of this solver neglects the
        // contribution of matrix Hsp in the mathematical formulation! There
        // could have other rangespace solvers in which both Hsp and Hps are
        // considered or neglected (in which case, only Vpp considered).
        //======================================================================

        // Setting the 1st column of M with dimension nbe (corresponding to ybe)
        Mbebe.noalias() = Ibebe + diag(Hbebe) * Tbebe;
        Mbibe.noalias() = -Tbibe;
        Mnibe.noalias() = tr(Sbeni);
        Mnpbe.noalias() = -Vnpbe * diag(inv(Hbebe)) - Vnpne * Bnebe;

        // Setting the 2nd column of M with dimension nbi (corresponding to xbi)
        Mbebi.noalias() = diag(-Hbebe) * Tbebi * diag(Hbibi);
        Mbibi.noalias() = Ibibi + Tbibi * diag(Hbibi);
        Mnibi.noalias() = -tr(Sbini) * diag(Hbibi);
        Mnpbi.noalias() = Vnpbi + Vnpne * Bnebi * diag(Hbibi);

        // Setting the 3rd column of M with dimension nni (corresponding to xni)
        Mbeni.noalias() = diag(-Hbebe) * Sbeni;
        Mbini.noalias() = Sbini;
        Mnini           = diag(Hnini);
        Mnpni.noalias() = Vnpni;

        // Setting the 4th column of M with dimension np (corresponding to p)
        Mbenp.noalias() = diag(-Hbebe) * Sbenp;
        Mbinp.noalias() = Sbinp;
        Mninp.fill(0.0);
        Mnpnp.noalias() = Vnpnp;

        lu.decompose(M);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        // The dimension variables needed below
        const auto ns  = args.dims.ns;
        const auto np  = args.dims.np;
        const auto nbs = args.dims.nbs;
        const auto nns = args.dims.nns;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;

        // View in Hd for the Hs entries in the diagonal of Hss
        const auto Hs = Hd.head(ns);

        // Views to the sub-matrices of the canonical matrix S
        const auto Sbsns = args.Sbsns;
        const auto Sbene = Sbsns.topLeftCorner(nbe, nne);
        const auto Sbine = Sbsns.bottomLeftCorner(nbi, nne);
        const auto Sbini = Sbsns.bottomRightCorner(nbi, nni);

        // Views to the sub-vectors of Hs = [Hbebe Hbibi Hnene Hnini]
        const auto Hbsbs = Hs.head(nbs);
        const auto Hnsns = Hs.tail(nns);
        const auto Hbebe = Hbsbs.head(nbe);
        const auto Hbibi = Hbsbs.tail(nbi);
        const auto Hnene = Hnsns.head(nne);

        // View to the sub-matrix Vnpnp === Vpp
        const auto Vnpnp = args.Hpp;

        // View to the sub-matrices Vnpbs = [Vnpbe Vnpbi]
        const auto Vnpbs = args.Hps.leftCols(nbs);
        const auto Vnpbe = Vnpbs.leftCols(nbe);
        const auto Vnpbi = Vnpbs.rightCols(nbi);

        // View to the sub-matrices Vnpns = [Vnpne Vnpni]
        const auto Vnpns = args.Hps.rightCols(nns);
        const auto Vnpne = Vnpns.leftCols(nne);
        const auto Vnpni = Vnpns.rightCols(nni);

        // The auxiliary matrix Bnebs = inv(Hnene) * tr(Sbsne) and its submatrices
        auto Bnebs = Bw.topLeftCorner(nne, nbs);
        auto Bnebe = Bnebs.leftCols(nbe);
        auto Bnebi = Bnebs.rightCols(nbi);

        // The auxiliary matrix Tbsbs = Sbsne * Bnebs and its submatrices
        auto Tbsbs = Tw.topLeftCorner(nbs, nbs);
        auto Tbibi = Tbsbs.bottomRightCorner(nbi, nbi);
        auto Tbibe = Tbsbs.bottomLeftCorner(nbi, nbe);
        auto Tbebi = Tbsbs.topRightCorner(nbe, nbi);
        auto Tbebe = Tbsbs.topLeftCorner(nbe, nbe);

        // Use ax as workspace for vector as
        auto as = ax.head(ns);

        // Use bb as workspace for vector bbs
        auto bbs = bb.head(nbs);

        // Views to the sub-vectors of as = [abs, ans]
        auto abs = as.head(nbs);
        auto ans = as.tail(nns);

        // Views to the sub-vectors of abs = [abe, abi]
        auto abe = abs.head(nbe);
        auto abi = abs.tail(nbi);

        // Views to the sub-vectors of ans = [ane, ani]
        auto ane = ans.head(nne);
        auto ani = ans.tail(nni);

        // Views to the sub-vectors in bbs = [bbe, bbi]
        auto bbe = bbs.head(nbe);
        auto bbi = bbs.tail(nbi);

        // Initialize vectors as, ap and bbs
        as = args.as;
        ap = args.ap;
        bbs = args.bbs;

        abe.noalias() = abe/Hbebe;
        ane.noalias() = ane/Hnene;

        ap.noalias() = ap - Vnpbe*abe - Vnpne*ane + Vnpne*Bnebi*abi;

        bbe.noalias() = bbe - Sbene*ane + Tbebi*abi - abe;
        bbi.noalias() = bbi - Sbine*ane + Tbibi*abi;
        ani.noalias() = ani - tr(Sbini) * abi;

        bbe.noalias()  = diag(-Hbebe)*bbe;

        auto r = rw.head(nbe + nbi + nni + np);

        auto ybe = r.head(nbe);
        auto xbi = r.segment(nbe, nbi);
        auto xni = r.segment(nbe + nbi, nni);
        auto xnp = r.segment(nbe + nbi + nni, np);

        r << bbe, bbi, ani, ap;

        lu.solve(r);

        const auto rank = lu.rank();

        assert(r.head(rank).allFinite());

        auto ybi = bbi;
        auto xbe = abe;
        auto xne = ane;

        ybi.noalias() = abi - diag(Hbibi)*xbi;
        xbe.noalias() = abe - ybe/Hbebe;
        xne.noalias() = ane - Bnebe*ybe - Bnebi*ybi;

        args.xs << xbe, xbi, xne, xni;
        args.p = xnp;
        args.ybs << ybe, ybi;
    }
};

SaddlePointSolverRangespace::SaddlePointSolverRangespace(Index nx, Index np, Index m)
: pimpl(new Impl(nx, np, m))
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
