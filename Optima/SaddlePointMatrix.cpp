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

// Eigen includes
using namespace Eigen;

namespace Optima {

SaddlePointMatrix::SaddlePointMatrix(MatrixXdConstRef H, MatrixXdConstRef A, Index nf)
: H(H), A(A), G(MatrixXd::Zero(0, 0)), nf(nf)
{}

SaddlePointMatrix::SaddlePointMatrix(MatrixXdConstRef H, MatrixXdConstRef A, MatrixXdConstRef G, Index nf)
: H(H), A(A), G(G), nf(nf)
{}

SaddlePointMatrix::operator MatrixXd() const
{
    const auto m = A.rows();
    const auto n = A.cols();
    const auto t = m + n;

    const auto nx = n - nf;
    const auto Ax = A.leftCols(nx);

    MatrixXd res = zeros(t, t);
    if(H.size()) res.topLeftCorner(nx, nx) = H.topLeftCorner(nx, nx);
    if(G.size()) res.bottomRightCorner(m, m) = G;
    res.block(nx, nx, nf, nf).diagonal().fill(1.0);
    res.topRightCorner(nx, m) = tr(Ax);
    res.bottomLeftCorner(m, n) = A;

    return res;
}

SaddlePointVector::SaddlePointVector(VectorXdConstRef a, VectorXdConstRef b)
: a(a), b(b)
{}

SaddlePointVector::SaddlePointVector(VectorXdConstRef r, Index n, Index m)
: a(r.head(n)), b(r.tail(m))
{}

SaddlePointVector::operator VectorXd() const
{
    const auto n = a.size();
    const auto m = b.size();
    const auto t = m + n;
    VectorXd res(t);
    res << a, b;
    return res;
}

SaddlePointSolution::SaddlePointSolution(VectorXdRef x, VectorXdRef y)
: x(x), y(y)
{}

SaddlePointSolution::SaddlePointSolution(VectorXdRef s, Index n, Index m)
: x(s.head(n)), y(s.tail(m))
{}

auto SaddlePointSolution::operator=(VectorXdConstRef vec) -> SaddlePointSolution&
{
    x.noalias() = vec.head(x.rows());
    y.noalias() = vec.tail(y.rows());
    return *this;
}

SaddlePointSolution::operator VectorXd() const
{
    const auto n = x.size();
    const auto m = y.size();
    const auto t = m + n;
    VectorXd res(t);
    res << x, y;
    return res;
}

auto operator*(SaddlePointMatrix lhs, VectorXdConstRef rhs) -> VectorXd
{
    MatrixXd M(lhs);
    VectorXd res = M * rhs;
    return res;
}

} // namespace Optima
