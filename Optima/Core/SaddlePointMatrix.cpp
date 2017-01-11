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

auto SaddlePointMatrix::dim() const -> Index
{
    return m_A.rows() + m_A.cols();
}

auto SaddlePointMatrix::H() const -> ConstMatrixRef
{
    return m_H;
}

auto SaddlePointMatrix::A() const -> ConstMatrixRef
{
    return m_A;
}

auto SaddlePointMatrix::fixed() const -> const Indices&
{
    return m_fixed.value();
}

auto SaddlePointMatrix::matrix() const -> MatrixXd
{
    const Index t = dim();
    MatrixXd res = zeros(t, t);
    res << *this;
    return res;
}

auto operator<<(MatrixRef mat, const SaddlePointMatrix& lhs) -> MatrixRef
{
    const auto& H = lhs.H();
    const auto& A = lhs.A();
    const auto& fixed = lhs.fixed();
    const Index n = A.cols();
    const Index m = A.rows();
    mat.topLeftCorner(n, n) << H;
    mat.topRightCorner(n, m) = tr(A);
    mat.bottomLeftCorner(m, n) = A;
    rows(mat, fixed).fill(0.0);
    for(Index i : fixed)
        mat(i, i) = 1.0;
    return mat;
}

auto operator<<(VectorRef vec, const SaddlePointVector& rhs) -> VectorRef
{
    const auto& a = rhs.a();
    const auto& b = rhs.b();
    const Index n = a.cols();
    const Index m = b.rows();
    vec.head(n).noalias() = a;
    vec.tail(m).noalias() = b;
    return vec;
}

auto operator*(const SaddlePointMatrix& lhs, ConstVectorRef rhs) -> VectorXd
{
    return lhs.matrix() * rhs;
}

} // namespace Optima
