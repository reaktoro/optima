// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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

#include "SaddlePointMatrix.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Utils.hpp>
using namespace Eigen;

namespace Optima {

SaddlePointMatrix::SaddlePointMatrix(HessianMatrixConstRef H, MatrixConstRef A, Index nf)
: SaddlePointMatrix(H, A, Matrix(), nf)
{}

SaddlePointMatrix::SaddlePointMatrix(HessianMatrixConstRef H, MatrixConstRef A, MatrixConstRef G, Index nf)
: H(H), A(A), G(G), nf(nf)
{
    Assert(isDenseMatrix(G) || isZeroMatrix(G),
        "Could not create a SaddlePointMatrix object.",
            "Matrix G must be either dense or an empty matrix.");

    Assert(H.dense.rows() == A.cols() || H.diagonal.rows() == A.cols(),
        "Could not create a SaddlePointMatrix object.",
            "Matrix A must have the same number of columns as there are rows in H.");

    Assert(A.rows() == G.rows() || G.size() == 0,
        "Could not create a SaddlePointMatrix object.",
            "Matrix G, when non-zero, must have the same number of rows and columns as there are rows in A.");
}

SaddlePointMatrix::operator Matrix() const
{
    const auto m = A.rows();
    const auto n = A.cols();
    const auto t = m + n;

    const auto nx = n - nf;
    const auto Ax = A.leftCols(nx);

    Matrix res = zeros(t, t);

    res.topLeftCorner(nx, nx) << H.topLeftCorner(nx);
    res.block(nx, nx, nf, nf).diagonal().fill(1.0);
    res.topRightCorner(nx, m) = tr(Ax);
    res.bottomLeftCorner(m, n) = A;
    if(G.size()) res.bottomRightCorner(m, m) = G;

    return res;
}

SaddlePointVector::SaddlePointVector(VectorConstRef a, VectorConstRef b)
: a(a), b(b)
{}

SaddlePointVector::SaddlePointVector(VectorConstRef r, Index n, Index m)
: a(r.head(n)), b(r.tail(m))
{}

SaddlePointVector::operator Vector() const
{
    const auto n = a.size();
    const auto m = b.size();
    const auto t = m + n;
    Vector res(t);
    res << a, b;
    return res;
}

SaddlePointSolution::SaddlePointSolution(VectorRef x, VectorRef y)
: x(x), y(y)
{}

SaddlePointSolution::SaddlePointSolution(VectorRef s, Index n, Index m)
: x(s.head(n)), y(s.tail(m))
{}

auto SaddlePointSolution::operator=(VectorConstRef vec) -> SaddlePointSolution&
{
    x.noalias() = vec.head(x.rows());
    y.noalias() = vec.tail(y.rows());
    return *this;
}

SaddlePointSolution::operator Vector() const
{
    const auto n = x.size();
    const auto m = y.size();
    const auto t = m + n;
    Vector res(t);
    res << x, y;
    return res;
}

auto operator*(SaddlePointMatrix lhs, VectorConstRef rhs) -> Vector
{
    Matrix M(lhs);
    Vector res = M * rhs;
    return res;
}

} // namespace Optima
