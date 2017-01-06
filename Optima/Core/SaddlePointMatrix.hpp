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
#include <Optima/Common/Index.hpp>
#include <Optima/Common/Optional.hpp>
#include <Optima/Core/HessianMatrix.hpp>
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// Used to represent the coefficient matrix in a saddle point problem.
/// A saddle point matrix is defined as a matrix with the following structure:
/// \f[
/// \begin{bmatrix}H & A^{T}\\ A & 0 \end{bmatrix}\begin{bmatrix}x\\ y \end{bmatrix}=\begin{bmatrix}a\\ b \end{bmatrix},
/// \f]
/// where \eq{H} is the *Hessian matrix* of an objective function and \eq{A} is
/// the *Jacobian matrix* of a constraint function. If the saddle point problem has fixed variables,
/// then the saddle point matrix has the following representation:
/// \f[
/// \begin{bmatrix}H_{\mathrm{x}} & 0 & A_{\mathrm{x}}^{T}\\ 0 & I_{\mathrm{f}} & 0\\ A_{\mathrm{x}} & A_{\mathrm{f}} & 0 \end{bmatrix}\begin{bmatrix}x_{\mathrm{x}}\\ x_{\mathrm{f}}\\ y \end{bmatrix}=\begin{bmatrix}a_{\mathrm{x}}\\ a_{\mathrm{f}}\\ b \end{bmatrix},
/// \f]
/// where the subscripts \eq{\mathrm{x}} and \eq{\mathrm{f}} correspond to free and fixed variables, respectively.
class SaddlePointMatrix
{
public:
    /// Construct a SaddlePointMatrix instance with given Hessian and Jacobian matrices.
    SaddlePointMatrix(const HessianMatrix& H, const ConstMatrixRef& A);

    /// Construct a SaddlePointMatrix instance with given Hessian and Jacobian matrices, and indices of fixed variables.
    SaddlePointMatrix(const HessianMatrix& H, const ConstMatrixRef& A, const Indices& fixed);

    /// Return the dimension of the saddle point mathrm.
    auto dim() const -> Index;

    /// Return the Hessian matrix *H*.
    auto H() const -> HessianMatrix;

    /// Return the Jacobian matrix *A*.
    auto A() const -> ConstMatrixRef;

    /// Return the indices of the fixed variables.
    auto fixed() const -> const Indices&;

    /// Convert this SaddlePointMatrix instance into a Matrix instance.
    auto matrix() const -> MatrixXd;

private:
    /// The Hessian matrix \eq{H} in the saddle point matrix.
    HessianMatrix m_H;

    /// The Jacobian matrix \eq{A} in the saddle point matrix.
    ConstMatrixRef m_A;

    /// The indices of the fixed variables.
    Optional<Indices> m_fixed;
};

/// A type used to describe a saddle point right-hand side vector.
class SaddlePointVector
{
public:
    /// Construct a SaddlePointVector instance with given *a* and *b* vectors.
    /// @param a The saddle point right-hand side vector *a*.
    /// @param b The saddle point right-hand side vector *b*.
    SaddlePointVector(ConstVectorRef a, ConstVectorRef b) : m_a(a), m_b(b) {}

    /// Construct a SaddlePointVector instance with given right-hand side vector.
    /// @param r The right-hand side vector *r* = [*a* *b*].
    /// @param n The dimension of vector *a*.
    /// @param m The dimension of vector *b*.
    SaddlePointVector(ConstVectorRef r, Index n, Index m) : m_a(r.head(n)), m_b(r.tail(m)) {}

    /// Return the dimension of the saddle point vector.
    auto dim() const -> Index { return m_a.rows() + m_b.rows(); }

    /// Return the solution vector *a*.
    auto a() const -> ConstVectorRef { return m_a; }

    /// Return the solution vector *b*.
    auto b() const -> ConstVectorRef { return m_b; }

    /// Convert this SaddlePointVector instance into a Vector instance.
    auto vector() const -> VectorXd { VectorXd res(dim()); res << m_a, m_b; return res; }

private:
    /// The saddle-point solution vector *a*.
    ConstVectorRef m_a;

    /// The saddle-point solution vector *b*.
    ConstVectorRef m_b;
};

/// A type used to describe a saddle point solution vector.
class SaddlePointSolution
{
public:
    /// Construct a SaddlePointSolution instance with given *x* and *y* vectors.
    /// @param x The saddle point solution vector *x*.
    /// @param y The saddle point solution vector *y*.
    SaddlePointSolution(VectorRef x, VectorRef y) : m_x(x), m_y(y) {}

    /// Construct a SaddlePointSolution instance with given solution vector.
    /// @param s The solution vector *s* = [*x* *y*].
    /// @param n The dimension of vector *x*.
    /// @param m The dimension of vector *y*.
    SaddlePointSolution(VectorRef s, Index n, Index m) : m_x(s.head(n)), m_y(s.tail(m)) {}

    /// Return the dimension of the saddle point solution vector.
    auto dim() const -> Index { return m_x.rows() + m_y.rows(); }

    /// Return the solution vector *x*.
    auto x() -> VectorRef { return m_x; }

    /// Return the solution vector *y*.
    auto y() -> VectorRef { return m_y; }

    /// Convert this SaddlePointSolution instance into a Vector instance.
    auto vector() const -> VectorXd { VectorXd res(dim()); res << m_x, m_y; return res; }

private:
    /// The saddle-point solution vector *x*.
    VectorRef m_x;

    /// The saddle-point solution vector *y*.
    VectorRef m_y;
};

/// Assign a SaddlePointMatrix instance into a MatrixRef instance.
auto operator<<(MatrixRef mat, const SaddlePointMatrix& lhs) -> MatrixRef;

} // namespace Optima

