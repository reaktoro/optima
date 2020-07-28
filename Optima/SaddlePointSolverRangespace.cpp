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
    Vector aw;  ///< The workspace for the right-hand side vectors a and b
    Vector bw;  ///< The workspace for the right-hand side vectors a and b
    Matrix mat; ///< The matrix used as a workspace for the decompose and solve methods.
    Vector vec; ///< The vector used as a workspace for the decompose and solve methods
    Vector Hx;  ///< The diagonal entries in the Hxx matrix.
    LU lu;      ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverRangespace::Impl instance.
    Impl(Index n, Index m)
    : mat(n + m, n + m), vec(n + m)
    {}

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        // The dimension variables needed below
        const auto nbx = args.dims.nbx;
        const auto nnx = args.dims.nnx;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;

        // Update the auxiliary vector Hx with the diagonal entries of Hxx
        Hx = args.Hxx.diagonal();

        // Views to the sub-matrices of Sbxnx = [Sbene Sbeni; Sbine Sbini]
        auto Sbxnx = args.Sbxnx;
        auto Sbeni = Sbxnx.topRightCorner(nbe, nni);
        auto Sbini = Sbxnx.bottomRightCorner(nbi, nni);
        auto Sbxne = Sbxnx.leftCols(nne);

        // Views to the sub-vectors of Hx = [Hbibi Hbebe Hnini Hnene]
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hbebe = Hbxbx.head(nbe);
        auto Hbibi = Hbxbx.tail(nbi);
        auto Hnene = Hnxnx.head(nne);
        auto Hnini = Hnxnx.tail(nni);

        // The auxiliary matrix Tbxbx = Sbxne * Bnebx and its submatrices
        auto Tbxbx = mat.topRightCorner(nbx, nbx);
        auto Tbibi = Tbxbx.bottomRightCorner(nbi, nbi);
        auto Tbibe = Tbxbx.bottomLeftCorner(nbi, nbe);
        auto Tbebi = Tbxbx.topRightCorner(nbe, nbi);
        auto Tbebe = Tbxbx.topLeftCorner(nbe, nbe);

        // The auxiliary matrix Bnebx = inv(Hnene) * tr(Sbxne)
        auto Bnebx = mat.rightCols(nbx).middleRows(nbx, nne);

        // The identity matrices Ibebe and Ibibi
        const auto Ibebe = identity(nbe, nbe);
        const auto Ibibi = identity(nbi, nbi);

        // Initialize workspace with zeros
        mat.fill(0.0);

        // The matrix M of the system of linear equations
        auto M = mat.bottomRightCorner(nbe + nbi + nni, nbe + nbi + nni);

        auto Mbe = M.topRows(nbe);
        auto Mbi = M.middleRows(nbe, nbi);
        auto Mni = M.bottomRows(nni);

        auto Mbebe = Mbe.leftCols(nbe);
        auto Mbebi = Mbe.middleCols(nbe, nbi);
        auto Mbeni = Mbe.rightCols(nni);

        auto Mbibe = Mbi.leftCols(nbe);
        auto Mbibi = Mbi.middleCols(nbe, nbi);
        auto Mbini = Mbi.rightCols(nni);

        auto Mnibe = Mni.leftCols(nbe);
        auto Mnibi = Mni.middleCols(nbe, nbi);
        auto Mnini = Mni.rightCols(nni);

        // Computing the auxiliary matrix Bnebx = inv(Hnene) * tr(Sbxne)
        Bnebx.noalias() = diag(inv(Hnene)) * tr(Sbxne);

        // Computing the auxiliary matrix Tbxbx = Sbxne * Bnebx
        Tbxbx.noalias() = Sbxne * Bnebx;

        // Setting the 2nd column of M with dimension nbe (corresponding to ybe)
        Mbebe.noalias() = Ibebe + diag(Hbebe) * Tbebe;
        Mbibe.noalias() = -Tbibe;
        Mnibe.noalias() = tr(Sbeni);

        // Setting the 1st column of M with dimension nbi (corresponding to xbi)
        Mbebi.noalias() = diag(-Hbebe) * Tbebi * diag(Hbibi);
        Mbibi.noalias() = Ibibi + Tbibi * diag(Hbibi);
        Mnibi.noalias() = -tr(Sbini) * diag(Hbibi);

        // Setting the 3rd column of M with dimension nni (corresponding to xni)
        Mbeni.noalias() = diag(-Hbebe) * Sbeni;
        Mbini.noalias() = Sbini;
        Mnini           = diag(Hnini);

        // std::cout << "M = \n" << M << std::endl; // TODO: Clean these commented out lines of code.

        lu.decompose(M);

        // std::cout << "L(M) = \n" << Matrix(lu.matrixLU().triangularView<Eigen::UnitLower>()) << std::endl;
        // std::cout << "U(M) = \n" << Matrix(lu.matrixLU().triangularView<Eigen::Upper>()) << std::endl;
        // std::cout << std::endl;
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        // The dimension variables needed below
        const auto nbx = args.dims.nbx;
        const auto nnx = args.dims.nnx;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = args.Sbxnx;
        auto Sbene = Sbxnx.topLeftCorner(nbe, nne);
        auto Sbine = Sbxnx.bottomLeftCorner(nbi, nne);
        auto Sbinx = Sbxnx.bottomRows(nbi);

        // The diagonal entries in H of the free variables, with Hx = [Hbibi Hbebe Hnini Hnene]
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hbebe = Hbxbx.head(nbe);
        auto Hbibi = Hbxbx.tail(nbi);
        auto Hnene = Hnxnx.head(nne);

        // Update the vector aw (workspace for vector ax)
        aw = args.ax;

        // Views to the sub-vectors in ax = [abx, anx]
        auto abx = aw.head(nbx);
        auto anx = aw.tail(nnx);
        auto abe = abx.head(nbe);
        auto abi = abx.tail(nbi);
        auto ane = anx.head(nne);
        auto ani = anx.tail(nni);

        // Update the vector bw (workspace for vector bbx)
        bw = args.bbx;

        // Views to the sub-vectors in bbx
        auto bbx = bw.head(nbx);
        auto bbe = bbx.head(nbe);
        auto bbi = bbx.tail(nbi);

        anx.noalias() -= tr(Sbinx) * abi;
        ane.noalias()  = ane/Hnene;
        bbe.noalias() -= abe/Hbebe;
        bbe.noalias() -= Sbene * ane;
        bbi.noalias() -= Sbine * ane;
        bbe.noalias()  = diag(-Hbebe) * bbe;

        auto r = vec.head(nbe + nbi + nni);

        auto ybe = r.head(nbe);
        auto xbi = r.segment(nbe, nbi);
        auto xni = r.segment(nbe + nbi, nni);

        r << bbe, bbi, ani;

        // std::cout << "bbe = " << tr(bbe) << std::endl;  // TODO: Clean these commented out lines of code.
        // std::cout << "bbi = " << tr(bbi) << std::endl;
        // std::cout << "ani = " << tr(ani) << std::endl;
        // std::cout << std::endl;

        lu.solve(r);

        const auto rank = lu.rank();

        assert(r.head(rank).allFinite());

        abe.noalias()  = (abe - ybe)/Hbebe;
        bbi.noalias()  = (abi - Hbibi % xbi);
        ane.noalias() -= (tr(Sbene)*ybe + tr(Sbine)*(bbi - abi))/Hnene;

        ani.noalias() = xni;
        bbe.noalias() = ybe;
        abi.noalias() = xbi;

        args.xx = aw;
        args.ybx = bw;
    }
};

SaddlePointSolverRangespace::SaddlePointSolverRangespace(Index n, Index m)
: pimpl(new Impl(n, m))
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
