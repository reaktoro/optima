// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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

// Eigenx includes
using namespace Eigen;

namespace Optima {

SaddlePointMatrix::SaddlePointMatrix(ConstMatrixRef H, ConstMatrixRef A)
: m_H(H), m_A(A)
{}

SaddlePointMatrix::SaddlePointMatrix(ConstMatrixRef H, ConstMatrixRef A, const Indices& fixed)
: m_H(H), m_A(A), m_fixed(fixed)
{}

auto SaddlePointMatrix::hessian() const -> ConstMatrixRef
{
    return m_H;
}

auto SaddlePointMatrix::jacobian() const -> ConstMatrixRef
{
    return m_A;
}

auto SaddlePointMatrix::fixed() const -> const Indices&
{
    return m_fixed.value();
}

auto SaddlePointMatrix::matrix() const -> MatrixXd
{
    const auto& H = hessian();
    const auto& A = jacobian();
    const Index n = H.rows();
    const Index m = A.rows();
    const Index t = n + m;
    MatrixXd res = zeros(t, t);
    res.topLeftCorner(n, n).diagonal() = H;
    res.topRightCorner(n, m)           = tr(A);
    res.bottomLeftCorner(m, n)         = A;
    rows(res, fixed()).fill(0.0);
    for(Index i : fixed())
        res(i, i) = 1.0;
    return res;
}

auto SaddlePointVector::vector() const -> VectorXd
{
    const Index n = x.rows();
    const Index m = y.rows();
    const Index t = n + m;
    VectorXd res(t);
    res << x, y;
    return res;
}

auto operator*(const SaddlePointMatrix& mat, const SaddlePointVector& vec) -> SaddlePointVector
{
    const MatrixXd A = mat.matrix();
    const VectorXd x = vec.vector();
    const VectorXd b = A*x;
    SaddlePointVector res;
    res.x = b.head(mat.jacobian().cols());
    res.y = b.tail(mat.jacobian().rows());
    return res;
}

} // namespace Optima
