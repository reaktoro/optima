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

#pragma once

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/HessianMatrix.hpp>
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
    /// The Hessian matrix \eq{H} in the saddle point matrix.
    HessianMatrixConstRef H;

    /// The Jacobian matrix \eq{A} in the saddle point matrix.
    MatrixConstRef A;

    /// The negative semi-definite matrix \eq{G} in the saddle point matrix.
    MatrixConstRef G;

    /// The number of fixed variables.
    Index nf;

    /// Construct a SaddlePointMatrix instance.
    /// @param H The \eq{H} matrix in the saddle point equation.
    /// @param A The \eq{A} matrix in the saddle point equation.
    /// @param nf The number of fixed variables.
    SaddlePointMatrix(MatrixConstRef H, MatrixConstRef A, Index nf = 0);

    /// Construct a SaddlePointMatrix instance.
    /// @param H The \eq{H} matrix in the saddle point equation.
    /// @param A The \eq{A} matrix in the saddle point equation.
    /// @param G The \eq{G} matrix in the saddle point equation.
    /// @param nf The number of fixed variables.
    SaddlePointMatrix(MatrixConstRef H, MatrixConstRef A, MatrixConstRef G, Index nf = 0);

    /// Convert this SaddlePointMatrix instance into a Matrix instance.
    operator Matrix() const;
};

/// A type used to describe a saddle point right-hand side vector.
class SaddlePointVector
{
public:
    /// The saddle-point solution vector \eq{a}.
    VectorConstRef a;

    /// The saddle-point solution vector \eq{b}.
    VectorConstRef b;

    /// Construct a SaddlePointVector instance with given \eq{a} and \eq{b} vectors.
    /// @param a The saddle point right-hand side vector \eq{a}.
    /// @param b The saddle point right-hand side vector \eq{b}.
    SaddlePointVector(VectorConstRef a, VectorConstRef b);

    /// Construct a SaddlePointVector instance with given right-hand side vector.
    /// @param r The right-hand side vector \eq{r=\begin{bmatrix}a & b\end{bmatrix}}.
    /// @param n The dimension of vector \eq{a}.
    /// @param m The dimension of vector \eq{b}.
    SaddlePointVector(VectorConstRef r, Index n, Index m);

    /// Convert this SaddlePointVector instance into a Vector instance.
    operator Vector() const;
};

/// A type used to describe a saddle point solution vector.
class SaddlePointSolution
{
public:
    /// The saddle-point solution vector \eq{x}.
    VectorRef x;

    /// The saddle-point solution vector \eq{y}.
    VectorRef y;

    /// Construct a SaddlePointSolution instance with given \eq{x} and \eq{y} vectors.
    /// @param x The saddle point solution vector \eq{x}.
    /// @param y The saddle point solution vector \eq{y}.
    SaddlePointSolution(VectorRef x, VectorRef y);

    /// Construct a SaddlePointSolution instance with given solution vector.
    /// @param s The solution vector \eq{s=\begin{bmatrix}x & y\end{bmatrix}}.
    /// @param n The dimension of vector \eq{x}.
    /// @param m The dimension of vector \eq{y}.
    SaddlePointSolution(VectorRef s, Index n, Index m);

    /// Assign this SaddlePointSolution instance with a VectorConstRef instance.
    auto operator=(VectorConstRef vec) -> SaddlePointSolution&;

    /// Convert this SaddlePointSolution instance into a Vector instance.
    operator Vector() const;
};

/// Return the multiplication of a SaddlePointMatrix by a vector.
auto operator*(SaddlePointMatrix lhs, VectorConstRef rhs) -> Vector;

} // namespace Optima

