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

// Eigen includes
#include <Optima/deps/eigen3/Eigen/src/LU/PartialPivLU.h>

// Optima includes
#include <Optima/Exception.hpp>
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

    Eigen::PartialPivLU<Matrix> lu; ///< The LU decomposition solver.

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
        const auto nb1 = args.dims.nb1;
        const auto nb2 = args.dims.nb2;
        const auto nn1 = args.dims.nn1;
        const auto nn2 = args.dims.nn2;

        // Update the auxiliary vector Hx with the diagonal entries of Hxx
        Hx = args.Hxx.diagonal();

        // Views to the sub-matrices of Sbxnx = [Sb1n1 Sb1n2; Sb2n1 Sb2n2]
        auto Sbxnx = args.Sbxnx;
        auto Sb1n2 = Sbxnx.topRightCorner(nb1, nn2);
        auto Sb2n2 = Sbxnx.bottomRightCorner(nb2, nn2);
        auto Sbxn1 = Sbxnx.leftCols(nn1);

        // Views to the sub-vectors of Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hb1b1 = Hbxbx.head(nb1);
        auto Hb2b2 = Hbxbx.tail(nb2);
        auto Hn1n1 = Hnxnx.head(nn1);
        auto Hn2n2 = Hnxnx.tail(nn2);

        // The auxiliary matrix Tbxbx = Sbxn1 * Bn1bx and its submatrices
        auto Tbxbx = mat.topRightCorner(nbx, nbx);
        auto Tb2b2 = Tbxbx.bottomRightCorner(nb2, nb2);
        auto Tb2b1 = Tbxbx.bottomLeftCorner(nb2, nb1);
        auto Tb1b2 = Tbxbx.topRightCorner(nb1, nb2);
        auto Tb1b1 = Tbxbx.topLeftCorner(nb1, nb1);

        // The auxiliary matrix Bn1bx = inv(Hn1n1) * tr(Sbxn1)
        auto Bn1bx = mat.rightCols(nbx).middleRows(nbx, nn1);

        // The identity matrices Ib1b1 and Ib2b2
        const auto Ib1b1 = identity(nb1, nb1);
        const auto Ib2b2 = identity(nb2, nb2);

        // Initialize workspace with zeros
        mat.fill(0.0);

        // The matrix M of the system of linear equations
        auto M = mat.bottomRightCorner(nb1 + nb2 + nn2, nb1 + nb2 + nn2);

        auto Mb1 = M.topRows(nb1);
        auto Mb2 = M.middleRows(nb1, nb2);
        auto Mn2 = M.bottomRows(nn2);

        auto Mb1b1 = Mb1.leftCols(nb1);
        auto Mb1b2 = Mb1.middleCols(nb1, nb2);
        auto Mb1n2 = Mb1.rightCols(nn2);

        auto Mb2b1 = Mb2.leftCols(nb1);
        auto Mb2b2 = Mb2.middleCols(nb1, nb2);
        auto Mb2n2 = Mb2.rightCols(nn2);

        auto Mn2b1 = Mn2.leftCols(nb1);
        auto Mn2b2 = Mn2.middleCols(nb1, nb2);
        auto Mn2n2 = Mn2.rightCols(nn2);

        // Computing the auxiliary matrix Bn1bx = inv(Hn1n1) * tr(Sbxn1)
        Bn1bx.noalias() = diag(inv(Hn1n1)) * tr(Sbxn1);

        // Computing the auxiliary matrix Tbxbx = Sbxn1 * Bn1bx
        Tbxbx.noalias() = Sbxn1 * Bn1bx;

        // Setting the 2nd column of M with dimension nb1 (corresponding to yb1)
        Mb1b1.noalias() = Ib1b1 + diag(Hb1b1) * Tb1b1;
        Mb2b1.noalias() = -Tb2b1;
        Mn2b1.noalias() = tr(Sb1n2);

        // Setting the 1st column of M with dimension nb2 (corresponding to xb2)
        Mb1b2.noalias() = diag(-Hb1b1) * Tb1b2 * diag(Hb2b2);
        Mb2b2.noalias() = Ib2b2 + Tb2b2 * diag(Hb2b2);
        Mn2b2.noalias() = -tr(Sb2n2) * diag(Hb2b2);

        // Setting the 3rd column of M with dimension nn2 (corresponding to xn2)
        Mb1n2.noalias() = diag(-Hb1b1) * Sb1n2;
        Mb2n2.noalias() = Sb2n2;
        Mn2n2           = diag(Hn2n2);

        lu.compute(M);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        // The dimension variables needed below
        const auto nbx = args.dims.nbx;
        const auto nnx = args.dims.nnx;
        const auto nb1 = args.dims.nb1;
        const auto nb2 = args.dims.nb2;
        const auto nn1 = args.dims.nn1;
        const auto nn2 = args.dims.nn2;

        // Views to the sub-matrices of the canonical matrix S
        auto Sbxnx = args.Sbxnx;
        auto Sb1n1 = Sbxnx.topLeftCorner(nb1, nn1);
        auto Sb2n1 = Sbxnx.bottomLeftCorner(nb2, nn1);
        auto Sb2nx = Sbxnx.bottomRows(nb2);

        // The diagonal entries in H of the free variables, with Hx = [Hb2b2 Hb1b1 Hn2n2 Hn1n1]
        auto Hbxbx = Hx.head(nbx);
        auto Hnxnx = Hx.tail(nnx);
        auto Hb1b1 = Hbxbx.head(nb1);
        auto Hb2b2 = Hbxbx.tail(nb2);
        auto Hn1n1 = Hnxnx.head(nn1);

        // Update the vector aw (workspace for vector ax)
        aw = args.ax;

        // Views to the sub-vectors in ax = [abx, anx]
        auto abx = aw.head(nbx);
        auto anx = aw.tail(nnx);
        auto ab1 = abx.head(nb1);
        auto ab2 = abx.tail(nb2);
        auto an1 = anx.head(nn1);
        auto an2 = anx.tail(nn2);

        // Update the vector bw (workspace for vector bbx)
        bw = args.bbx;

        // Views to the sub-vectors in bbx
        auto bbx = bw.head(nbx);
        auto bb1 = bbx.head(nb1);
        auto bb2 = bbx.tail(nb2);

        anx.noalias() -= tr(Sb2nx) * ab2;
        an1.noalias()  = an1/Hn1n1;
        bb1.noalias() -= ab1/Hb1b1;
        bb1.noalias() -= Sb1n1 * an1;
        bb2.noalias() -= Sb2n1 * an1;
        bb1.noalias()  = diag(-Hb1b1) * bb1;

        auto r = vec.head(nb1 + nb2 + nn2);

        auto yb1 = r.head(nb1);
        auto xb2 = r.segment(nb1, nb2);
        auto xn2 = r.segment(nb1 + nb2, nn2);

        r << bb1, bb2, an2;

        r.noalias() = lu.solve(r);

        ab1.noalias()  = (ab1 - yb1)/Hb1b1;
        bb2.noalias()  = (ab2 - Hb2b2 % xb2);
        an1.noalias() -= (tr(Sb1n1)*yb1 + tr(Sb2n1)*(bb2 - ab2))/Hn1n1;

        an2.noalias() = xn2;
        bb1.noalias() = yb1;
        ab2.noalias() = xb2;

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
