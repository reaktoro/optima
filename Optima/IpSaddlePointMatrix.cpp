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
    Index nx, Index nf)
: m_H(H), m_A(A), m_Z(Z), m_W(W), m_L(L), m_U(U), m_nx(nx), m_nf(nf) 
{}

auto IpSaddlePointMatrix::matrix() const -> MatrixXd
{
    const Index t = size();
    MatrixXd res = zeros(t, t);
    res << *this;
    return res;
}

IpSaddlePointVector::IpSaddlePointVector(
    VectorXdConstRef a,
    VectorXdConstRef b,
    VectorXdConstRef c,
    VectorXdConstRef d) : m_a(a), m_b(b), m_c(c), m_d(d)
{}

IpSaddlePointVector::IpSaddlePointVector(VectorXdConstRef r, Index n, Index m)
: m_a(r.head(n)),
  m_b(r.segment(n, m)),
  m_c(r.segment(n + m, n)),
  m_d(r.tail(n))
{}

auto IpSaddlePointVector::vector() const -> VectorXd
{
    VectorXd res(size());
    res << m_a, m_b, m_c, m_d;
    return res;
}

IpSaddlePointSolution::IpSaddlePointSolution(
    VectorXdRef x,
    VectorXdRef y,
    VectorXdRef z,
    VectorXdRef w)
: m_x(x), m_y(y), m_z(z), m_w(w)
{}

IpSaddlePointSolution::IpSaddlePointSolution(VectorXdRef s, Index n, Index m)
: m_x(s.head(n)),
  m_y(s.segment(n, m)),
  m_z(s.segment(n + m, n)),
  m_w(s.tail(n))
{}

auto IpSaddlePointSolution::operator=(VectorXdConstRef vec) -> IpSaddlePointSolution&
{
    const Index n = m_x.size();
    const Index m = m_y.size();
    m_x.noalias() = vec.head(n);
    m_y.noalias() = vec.segment(n, m);
    m_z.noalias() = vec.segment(n + m, n);
    m_w.noalias() = vec.tail(n);
    return *this;
}

auto IpSaddlePointSolution::vector() const -> VectorXd
{
    VectorXd res(size());
    res << m_x, m_y, m_z, m_w;
    return res;
}

auto operator<<(MatrixXdRef mat, const IpSaddlePointMatrix& lhs) -> MatrixXdRef
{
    const auto n  = lhs.n();
    const auto nx = lhs.nx();
    const auto nf = lhs.nf();
    const auto m  = lhs.m();
    const auto t  = lhs.size();
    const auto H  = lhs.H();
    const auto A  = lhs.A();
    const auto Z  = lhs.Z();
    const auto W  = lhs.W();
    const auto L  = lhs.L();
    const auto U  = lhs.U();
    
    const auto Hx = lhs.H().topLeftCorner(nx, nx);
    const auto Ax = lhs.A().leftCols(nx);
    const auto Zx = Z.head(nx);
    const auto Wx = W.head(nx);
    const auto Lx = L.head(nx);
    const auto Ux = U.head(nx);

    mat = zeros(t, t);
    mat.topRows(nx).leftCols(nx) = Hx;
    mat.topRows(nx).middleCols(n, m) = tr(Ax);
    mat.topRows(nx).middleCols(n + m, nx).diagonal().fill(-1.0);
    mat.topRows(nx).middleCols(n + m + n, nx).diagonal().fill(-1.0);
    mat.middleRows(nx, nf).middleCols(nx, nf).diagonal().fill(1.0);
    mat.middleRows(n, m).leftCols(n) = A;
    mat.middleRows(n + m, nx).leftCols(nx).diagonal() = Zx;
    mat.middleRows(n + m + n, nx).leftCols(nx).diagonal() = Wx;
    mat.bottomRightCorner(2*n, 2*n).diagonal() << Lx, ones(nf), Ux, ones(nf);
    return mat;
}

auto operator<<(VectorXdRef vec, const IpSaddlePointVector& rhs) -> VectorXdRef
{
    const auto a = rhs.a();
    const auto b = rhs.b();
    const auto c = rhs.c();
    const auto d = rhs.d();
    const Index n = a.rows();
    const Index m = b.rows();
    const Index t = 3*n + m;
    vec.resize(t);
    vec << a, b, c, d;
    return vec;
}

auto operator*(const IpSaddlePointMatrix& lhs, VectorXdConstRef rhs) -> VectorXd
{
    return lhs.matrix() * rhs;
}

} // namespace Optima
