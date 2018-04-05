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

#include "IpSaddlePointMatrix.hpp"

// Optima includes
#include <Optima/Utils.hpp>
using namespace Eigen;

namespace Optima {

IpSaddlePointMatrix::IpSaddlePointMatrix(
    VariantMatrixConstRef H,
    MatrixConstRef A,
    VectorConstRef Z,
    VectorConstRef W,
    VectorConstRef L,
    VectorConstRef U,
    VectorXiConstRef jf)
: H(H), A(A), Z(Z), W(W), L(L), U(U), jf(jf)
{}

IpSaddlePointMatrix::operator Matrix() const
{
    const auto m = A.rows();
    const auto n = A.cols();
    const auto t = 3*n + m;

    Matrix res = zeros(t, t);

    // Block: H
    res.topLeftCorner(n, n) << H;
    res.topLeftCorner(n, n)(all, jf).fill(0.0);
    res.topLeftCorner(n, n)(jf, all).fill(0.0);
    res.topLeftCorner(n, n).diagonal()(jf).fill(1.0);

    // Block: tr(A)
    res.middleCols(n, m).topRows(n) = tr(A);
    res.middleCols(n, m).topRows(n)(jf, all).fill(0.0);

    // Block: -I
    res.middleCols(n + m, n).diagonal().fill(-1.0);
    res.middleCols(n + m, n).diagonal()(jf).fill(0.0);

    // Block: -I
    res.rightCols(n).diagonal().fill(-1.0);
    res.rightCols(n).diagonal()(jf).fill(0.0);

    // Block: A
    res.middleRows(n, m).leftCols(n) = A;

    // Block: Z
    res.middleRows(n + m, n).leftCols(n).diagonal() = Z;
    res.middleRows(n + m, n).leftCols(n).diagonal()(jf).fill(0.0);

    // Block: L
    res.middleRows(n + m, n).middleCols(n + m, n).diagonal() = L;
    res.middleRows(n + m, n).middleCols(n + m, n).diagonal()(jf).fill(1.0);

    // Block: W
    res.bottomRows(n).leftCols(n).diagonal() = W;
    res.bottomRows(n).leftCols(n).diagonal()(jf).fill(0.0);

    // Block: U
    res.bottomRows(n).rightCols(n).diagonal() = U;
    res.bottomRows(n).rightCols(n).diagonal()(jf).fill(1.0);

    return res;
}

IpSaddlePointVector::IpSaddlePointVector(
    VectorConstRef x,
    VectorConstRef y,
    VectorConstRef z,
    VectorConstRef w)
: x(x), y(y), z(z), w(w),
  a(x), b(y), c(z), d(w)
{}

IpSaddlePointVector::IpSaddlePointVector(VectorConstRef r, Index n, Index m)
: x(r.head(n)),
  y(r.segment(n, m)),
  z(r.segment(n + m, n)),
  w(r.tail(n)),
  a(x), b(y), c(z), d(w)
{}

IpSaddlePointVector::operator Vector() const
{
    const auto n = x.size();
    const auto m = y.size();
    const auto t = 3*n + m;
    Vector res(t);
    res << x, y, z, w;
    return res;
}

IpSaddlePointSolution::IpSaddlePointSolution(
    VectorRef x,
    VectorRef y,
    VectorRef z,
    VectorRef w)
: x(x), y(y), z(z), w(w)
{}

IpSaddlePointSolution::IpSaddlePointSolution(VectorRef s, Index n, Index m)
: x(s.head(n)),
  y(s.segment(n, m)),
  z(s.segment(n + m, n)),
  w(s.tail(n))
{}

auto IpSaddlePointSolution::operator=(VectorConstRef vec) -> IpSaddlePointSolution&
{
    const auto n = x.size();
    const auto m = y.size();
    x.noalias() = vec.head(n);
    y.noalias() = vec.segment(n, m);
    z.noalias() = vec.segment(n + m, n);
    w.noalias() = vec.tail(n);
    return *this;
}

IpSaddlePointSolution::operator Vector() const
{
    const auto n = x.size();
    const auto m = y.size();
    const auto t = 3*n + m;
    Vector res(t);
    res << x, y, z, w;
    return res;
}

auto operator*(IpSaddlePointMatrix lhs, VectorConstRef rhs) -> Vector
{
    Matrix M(lhs);
    Vector res = M * rhs;
    return res;
}

} // namespace Optima
