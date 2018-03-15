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

// Eigen includes
using namespace Eigen;

namespace Optima {

IpSaddlePointMatrix::IpSaddlePointMatrix(
    MatrixXdConstRef H,
    MatrixXdConstRef A,
    VectorXdConstRef Z,
    VectorXdConstRef W,
    VectorXdConstRef L,
    VectorXdConstRef U,
    Index nf)
: H(H), A(A), Z(Z), W(W), L(L), U(U), nf(nf)
{}

IpSaddlePointMatrix::operator MatrixXd() const
{
    const auto m = A.rows();
    const auto n = A.cols();
    const auto t = 3*n + m;

    const auto nx = n - nf;
    const auto Hx = H.topLeftCorner(nx, nx);
    const auto Ax = A.leftCols(nx);
    const auto Zx = Z.head(nx);
    const auto Wx = W.head(nx);
    const auto Lx = L.head(nx);
    const auto Ux = U.head(nx);

    MatrixXd res = zeros(t, t);
    res.topRows(nx).leftCols(nx) = Hx;
    res.topRows(nx).middleCols(n, m) = tr(Ax);
    res.topRows(nx).middleCols(n + m, nx).diagonal().fill(-1.0);
    res.topRows(nx).middleCols(n + m + n, nx).diagonal().fill(-1.0);
    res.middleRows(nx, nf).middleCols(nx, nf).diagonal().fill(1.0);
    res.middleRows(n, m).leftCols(n) = A;
    res.middleRows(n + m, nx).leftCols(nx).diagonal() = Zx;
    res.middleRows(n + m + n, nx).leftCols(nx).diagonal() = Wx;
    res.bottomRightCorner(2*n, 2*n).diagonal() << Lx, ones(nf), Ux, ones(nf);

    return res;
}

IpSaddlePointVector::IpSaddlePointVector(
    VectorXdConstRef a,
    VectorXdConstRef b,
    VectorXdConstRef c,
    VectorXdConstRef d) : a(a), b(b), c(c), d(d)
{}

IpSaddlePointVector::IpSaddlePointVector(VectorXdConstRef r, Index n, Index m)
: a(r.head(n)),
  b(r.segment(n, m)),
  c(r.segment(n + m, n)),
  d(r.tail(n))
{}

IpSaddlePointVector::operator VectorXd() const
{
    const auto n = a.size();
    const auto m = b.size();
    const auto t = 3*n + m;
    VectorXd res(t);
    res << a, b, c, d;
    return res;
}

IpSaddlePointSolution::IpSaddlePointSolution(
    VectorXdRef x,
    VectorXdRef y,
    VectorXdRef z,
    VectorXdRef w)
: x(x), y(y), z(z), w(w)
{}

IpSaddlePointSolution::IpSaddlePointSolution(VectorXdRef s, Index n, Index m)
: x(s.head(n)),
  y(s.segment(n, m)),
  z(s.segment(n + m, n)),
  w(s.tail(n))
{}

auto IpSaddlePointSolution::operator=(VectorXdConstRef vec) -> IpSaddlePointSolution&
{
    const auto n = x.size();
    const auto m = y.size();
    x.noalias() = vec.head(n);
    y.noalias() = vec.segment(n, m);
    z.noalias() = vec.segment(n + m, n);
    w.noalias() = vec.tail(n);
    return *this;
}

IpSaddlePointSolution::operator VectorXd() const
{
    const auto n = x.size();
    const auto m = y.size();
    const auto t = 3*n + m;
    VectorXd res(t);
    res << x, y, z, w;
    return res;
}

auto operator*(IpSaddlePointMatrix lhs, VectorXdConstRef rhs) -> VectorXd
{
    MatrixXd M(lhs);
    VectorXd res = M * rhs;
    return res;
}

} // namespace Optima
