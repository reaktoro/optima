// Optima is a C++ library for solving linear and non-linear constrained optimization problems
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
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent the coefficient matrix in a saddle point problem.
/// A saddle point matrix is defined as a matrix with the following structure:
/// \f[
/// \begin{bmatrix}H + D & A\\ A & G \end{bmatrix}\begin{bmatrix}x \\ y \end{bmatrix}=\begin{bmatrix}a \\ b \end{bmatrix},
/// \f]
/// where \eq{H} is the *Hessian matrix* of an objective function, \eq{D} is an optional diagonal matrix
/// contribution, \eq{A} is the *Jacobian matrix* of a constraint function, and \eq{G} is a negative
/// semi-definite matrix.
/// If the saddle point problem has fixed variables, then the saddle point matrix has the following representation:
/// \f[
/// \begin{bmatrix}H_{\mathrm{x}} + D_{\mathrm{x}} & 0 & A_{\mathrm{x}}^{T}\\ 0 & I_{\mathrm{f}} & 0\\ A_{\mathrm{x}} & A_{\mathrm{f}} & G \end{bmatrix}\begin{bmatrix}x_{\mathrm{x}}\\ x_{\mathrm{f}}\\ y \end{bmatrix}=\begin{bmatrix}a_{\mathrm{x}}\\ a_{\mathrm{f}}\\ b \end{bmatrix},
/// \f]
/// where the subscripts \eq{\mathrm{x}} and \eq{\mathrm{f}} correspond to free and fixed variables, respectively.
class SaddlePointMatrix
{
public:
    /// The Hessian matrix \eq{H} in the saddle point matrix.
    MatrixConstRef H;

    /// The diagonal matrix \eq{D} in the saddle point matrix.
    VectorConstRef D;

    /// The top-block of the Jacobian matrix \eq{A} in the saddle point matrix that always remain constant.
    MatrixConstRef At;

    /// The bottom-block of the Jacobian matrix \eq{A} in the saddle point matrix that is variable.
    MatrixConstRef Ab;

    /// The negative semi-definite matrix \eq{G} in the saddle point matrix.
    MatrixConstRef G;

    /// The indices of the fixed variables.
    IndicesConstRef jf;

    /// Construct a SaddlePointMatrix instance.
    /// @param H The \eq{H} matrix in the saddle point matrix.
    /// @param D The \eq{D} matrix in the saddle point matrix.
    /// @param At The top-block of \eq{A} matrix in the saddle point matrix.
    /// @param Ab The bottom-block of \eq{A} matrix in the saddle point matrix.
    /// @param jf The indices of the fixed variables.
    SaddlePointMatrix(MatrixConstRef H, VectorConstRef D, MatrixConstRef At, MatrixConstRef Ab, IndicesConstRef jf);

    /// Construct a SaddlePointMatrix instance.
    /// @param H The \eq{H} matrix in the saddle point matrix.
    /// @param D The \eq{D} matrix in the saddle point matrix.
    /// @param At The top-block of \eq{A} matrix in the saddle point matrix.
    /// @param Ab The bottom-block of \eq{A} matrix in the saddle point matrix.
    /// @param G The \eq{G} matrix in the saddle point matrix.
    /// @param jf The indices of the fixed variables.
    SaddlePointMatrix(MatrixConstRef H, VectorConstRef D, MatrixConstRef At, MatrixConstRef Ab, MatrixConstRef G, IndicesConstRef jf);

    /// Convert this SaddlePointMatrix instance into a Matrix instance.
    operator Matrix() const;
};

/// Used to describe a saddle point vector \eq{\begin{bmatrix}x & y\end{bmatrix}}.
class SaddlePointVector
{
public:
    /// The saddle point sub-vector \eq{x}.
    VectorConstRef x;

    /// The saddle point sub-vector \eq{y}.
    VectorConstRef y;

    /// The saddle point right-hand side sub-vector \eq{a} *(this is an alias to x)*.
    VectorConstRef a;

    /// The saddle point right-hand side sub-vector \eq{b} *(this is an alias to y)*.
    VectorConstRef b;

    /// Construct a SaddlePointVector instance with given \eq{x} and \eq{y} sub-vectors.
    /// @param x The saddle point sub-vector \eq{x}.
    /// @param y The saddle point sub-vector \eq{y}.
    SaddlePointVector(VectorConstRef x, VectorConstRef y);

    /// Construct a SaddlePointVector instance with given vector.
    /// @param r The vector \eq{r=\begin{bmatrix}x & y\end{bmatrix}}.
    /// @param n The dimension of sub-vector \eq{x}.
    /// @param m The dimension of sub-vector \eq{y}.
    SaddlePointVector(VectorConstRef r, Index n, Index m);

    /// Convert this SaddlePointVector instance into a Vector instance.
    operator Vector() const;
};

/// Used to describe a saddle point solution vector \eq{\begin{bmatrix}x & y\end{bmatrix}}.
class SaddlePointSolution
{
public:
    /// The solution sub-vector \eq{x} in the saddle point problem.
    VectorRef x;

    /// The solution sub-vector \eq{y} in the saddle point problem.
    VectorRef y;

    /// Construct a SaddlePointSolution instance with given \eq{x} and \eq{y} sub-vectors.
    /// @param x The saddle point sub-vector \eq{x}.
    /// @param y The saddle point sub-vector \eq{y}.
    SaddlePointSolution(VectorRef x, VectorRef y);

    /// Construct a SaddlePointSolution instance with given vector.
    /// @param r The vector \eq{r=\begin{bmatrix}x & y\end{bmatrix}}.
    /// @param n The dimension of sub-vector \eq{x}.
    /// @param m The dimension of sub-vector \eq{y}.
    SaddlePointSolution(VectorRef r, Index n, Index m);

    /// Assign a VectorConstRef instance to this SaddlePointSolution instance.
    auto operator=(VectorConstRef vec) -> SaddlePointSolution&;

    /// Convert this SaddlePointSolution instance into a Vector instance.
    operator Vector() const;
};

/// Return the multiplication of a SaddlePointMatrix by a vector.
auto operator*(SaddlePointMatrix lhs, VectorConstRef rhs) -> Vector;

} // namespace Optima

