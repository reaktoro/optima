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

SaddlePointMatrix::SaddlePointMatrix(MatrixXdConstRef H, MatrixXdConstRef A, MatrixXdConstRef G, Index nx, Index nf)
: m_H(H), m_A(A), m_G(G), m_nx(nx), m_nf(nf)
{}

auto SaddlePointMatrix::H() const -> MatrixXdConstRef
{
    return m_H;
}

auto SaddlePointMatrix::A() const -> MatrixXdConstRef
{
    return m_A;
}

auto SaddlePointMatrix::G() const -> MatrixXdConstRef
{
    return m_G;
}

auto SaddlePointMatrix::size() const -> Index
{
    return m_A.rows() + m_A.cols();
}

auto SaddlePointMatrix::n() const -> Index
{
    return nx() + nf();
}

auto SaddlePointMatrix::m() const -> Index
{
    return m_A.rows();
}

auto SaddlePointMatrix::nx() const -> Index
{
    return m_nx;
}

auto SaddlePointMatrix::nf() const -> Index
{
    return m_nf;
}

auto SaddlePointMatrix::matrix() const -> MatrixXd
{
    const Index t = size();
    MatrixXd res = zeros(t, t);
    res << *this;
    return res;
}

auto operator<<(MatrixXdRef mat, const SaddlePointMatrix& lhs) -> MatrixXdRef
{
    const auto nx = lhs.nx();
    const auto nf = lhs.nf();
    const auto H = lhs.H();
    const auto A = lhs.A();
    const auto Ax = lhs.A().leftCols(nx);
    const auto G = lhs.G();
    const auto n = lhs.n();
    const auto m = lhs.m();
    mat.fill(0.0);
    if(H.size()) mat.topLeftCorner(nx, nx) << H.topLeftCorner(nx, nx);
    if(G.size()) mat.bottomRightCorner(m, m) = G;
    mat.block(nx, nx, nf, nf).diagonal().fill(1.0);
    mat.topRightCorner(nx, m) = tr(Ax);
    mat.bottomLeftCorner(m, n) = A;
    return mat;
}

auto operator<<(VectorXdRef vec, const SaddlePointVector& rhs) -> VectorXdRef
{
    const auto& a = rhs.a();
    const auto& b = rhs.b();
    const Index n = a.cols();
    const Index m = b.rows();
    vec.head(n).noalias() = a;
    vec.tail(m).noalias() = b;
    return vec;
}

auto operator*(const SaddlePointMatrix& lhs, VectorXdConstRef rhs) -> VectorXd
{
    return lhs.matrix() * rhs;
}

} // namespace Optima
