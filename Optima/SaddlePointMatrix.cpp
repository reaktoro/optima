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
#include <Optima/Utils.hpp>
using namespace Eigen;

namespace Optima {

SaddlePointMatrix::SaddlePointMatrix(MatrixConstRef H, MatrixConstRef A, Index nf)
: H(H), A(A), G(zeros(0, 0)), nf(nf)
{
}

SaddlePointMatrix::SaddlePointMatrix(MatrixConstRef H, MatrixConstRef A, MatrixConstRef G, Index nf)
: H(H), A(A), G(G), nf(nf)
{}

SaddlePointMatrix::operator Matrix() const
{
    const auto m = A.rows();
    const auto n = A.cols();
    const auto t = m + n;

    const auto nx = n - nf;
    const auto Ax = A.leftCols(nx);

    Matrix res = zeros(t, t);
    if(isDenseMatrix(H)) res.topLeftCorner(nx, nx) = H.topLeftCorner(nx, nx);
    if(isDiagonalMatrix(H)) res.topLeftCorner(nx, nx).diagonal() = H.col(0).head(nx);
    if(isDenseMatrix(G)) res.bottomRightCorner(m, m) = G;
    if(isDiagonalMatrix(G)) res.bottomRightCorner(m, m).diagonal() = G.col(0);
    res.block(nx, nx, nf, nf).diagonal().fill(1.0);
    res.topRightCorner(nx, m) = tr(Ax);
    res.bottomLeftCorner(m, n) = A;

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
