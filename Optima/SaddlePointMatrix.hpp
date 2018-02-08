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

#pragma once

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Optional.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent the coefficient matrix in a saddle point problem.
/// A saddle point matrix is defined as a matrix with the following structure:
/// \f[
/// \begin{bmatrix}H & A\\ A & G \end{bmatrix}\begin{bmatrix}x \\ y \end{bmatrix}=\begin{bmatrix}a \\ b \end{bmatrix},
/// \f]
/// where \eq{H} is the *Hessian matrix* of an objective function, \eq{A} is the *Jacobian matrix*
/// of a constraint function, and \eq{G} is a negative semi-definite matrix.
/// If the saddle point problem has fixed variables, then the saddle point matrix has the following
/// representation:
/// \f[
/// \begin{bmatrix}H_{\mathrm{x}} & 0 & A_{\mathrm{x}}^{T}\\ 0 & I_{\mathrm{f}} & 0\\ A_{\mathrm{x}} & A_{\mathrm{f}} & G \end{bmatrix}\begin{bmatrix}x_{\mathrm{x}}\\ x_{\mathrm{f}}\\ y \end{bmatrix}=\begin{bmatrix}a_{\mathrm{x}}\\ a_{\mathrm{f}}\\ b \end{bmatrix},
/// \f]
/// where the subscripts \eq{\mathrm{x}} and \eq{\mathrm{f}} correspond to free and fixed variables, respectively.
class SaddlePointMatrix
{
public:
    /// Construct a SaddlePointMatrix instance.
    /// @param H The \eq{H} matrix in the saddle point equation.
    /// @param A The \eq{A} matrix in the saddle point equation.
    /// @param G The \eq{G} matrix in the saddle point equation.
    /// @param nx The number of free variables.
    /// @param nf The number of fixed variables.
    SaddlePointMatrix(MatrixXdConstRef H, MatrixXdConstRef A, MatrixXdConstRef G, Index nx, Index nf = 0);

    /// Return the Hessian matrix *H*.
    auto H() const -> MatrixXdConstRef;

    /// Return the Jacobian matrix *A*.
    auto A() const -> MatrixXdConstRef;

    /// Return the matrix *G*.
    auto G() const -> MatrixXdConstRef;

    /// Return the dimension of the saddle point matrix.
    auto size() const -> Index;

    /// Return the number of variables.
    auto n() const -> Index;

    /// Return the number of linear equality constraints.
    auto m() const -> Index;

    /// Return the number of free variables.
    auto nx() const -> Index;

    /// Return the number of fixed variables.
    auto nf() const -> Index;

    /// Convert this SaddlePointMatrix instance into a Matrix instance.
    auto matrix() const -> MatrixXd;

private:
    /// The Hessian matrix \eq{H} in the saddle point matrix.
    MatrixXdConstRef m_H;

    /// The Jacobian matrix \eq{A} in the saddle point matrix.
    MatrixXdConstRef m_A;

    /// The negative semi-definite matrix \eq{G} in the saddle point matrix.
    MatrixXdConstRef m_G;

    /// The number of free variables.
    Index m_nx;

    /// The number of fixed variables.
    Index m_nf;
};

/// A type used to describe a saddle point right-hand side vector.
class SaddlePointVector
{
public:
    /// Construct a SaddlePointVector instance with given *a* and *b* vectors.
    /// @param a The saddle point right-hand side vector *a*.
    /// @param b The saddle point right-hand side vector *b*.
    SaddlePointVector(VectorXdConstRef a, VectorXdConstRef b) : m_a(a), m_b(b) {}

    /// Construct a SaddlePointVector instance with given right-hand side vector.
    /// @param r The right-hand side vector *r* = [*a* *b*].
    /// @param n The dimension of vector *a*.
    /// @param m The dimension of vector *b*.
    SaddlePointVector(VectorXdConstRef r, Index n, Index m) : m_a(r.head(n)), m_b(r.tail(m)) {}

    /// Return the dimension of the saddle point vector.
    auto dim() const -> Index { return m_a.rows() + m_b.rows(); }

    /// Return the solution vector *a*.
    auto a() const -> VectorXdConstRef { return m_a; }

    /// Return the solution vector *b*.
    auto b() const -> VectorXdConstRef { return m_b; }

    /// Convert this SaddlePointVector instance into a Vector instance.
    auto vector() const -> VectorXd { VectorXd res(dim()); res << m_a, m_b; return res; }

private:
    /// The saddle-point solution vector *a*.
    VectorXdConstRef m_a;

    /// The saddle-point solution vector *b*.
    VectorXdConstRef m_b;
};

/// A type used to describe a saddle point solution vector.
class SaddlePointSolution
{
public:
    /// Construct a SaddlePointSolution instance with given *x* and *y* vectors.
    /// @param x The saddle point solution vector *x*.
    /// @param y The saddle point solution vector *y*.
    SaddlePointSolution(VectorXdRef x, VectorXdRef y) : m_x(x), m_y(y) {}

    /// Construct a SaddlePointSolution instance with given solution vector.
    /// @param s The solution vector *s* = [*x* *y*].
    /// @param n The dimension of vector *x*.
    /// @param m The dimension of vector *y*.
    SaddlePointSolution(VectorXdRef s, Index n, Index m) : m_x(s.head(n)), m_y(s.tail(m)) {}

    /// Assign this SaddlePointSolution instance with a VectorXdConstRef instance.
    auto operator=(VectorXdConstRef vec) -> SaddlePointSolution& { m_x.noalias() = vec.head(m_x.rows()); m_y.noalias() = vec.tail(m_y.rows()); return *this; }

    /// Return the dimension of the saddle point solution vector.
    auto dim() const -> Index { return m_x.rows() + m_y.rows(); }

    /// Return the solution vector *x*.
    auto x() -> VectorXdRef { return m_x; }

    /// Return the solution vector *y*.
    auto y() -> VectorXdRef { return m_y; }

    /// Convert this SaddlePointSolution instance into a Vector instance.
    auto vector() const -> VectorXd { VectorXd res(dim()); res << m_x, m_y; return res; }

private:
    /// The saddle-point solution vector *x*.
    VectorXdRef m_x;

    /// The saddle-point solution vector *y*.
    VectorXdRef m_y;
};

/// Assign a MatrixXdRef instance with a SaddlePointMatrix instance.
auto operator<<(MatrixXdRef mat, const SaddlePointMatrix& lhs) -> MatrixXdRef;

/// Assign a VectorXdRef instance with a SaddlePointVector instance.
auto operator<<(VectorXdRef vec, const SaddlePointVector& rhs) -> VectorXdRef;

/// Return the multiplication of a SaddlePointMatrix by a vector.
auto operator*(const SaddlePointMatrix& lhs, VectorXdConstRef rhs) -> VectorXd;

} // namespace Optima

