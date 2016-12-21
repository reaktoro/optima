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

// C++ includes
#include <map>

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// Used to represent the coefficient matrix in a saddle point problem.
/// A saddle point matrix is defined as a matrix with the following structure:
/// \f[
/// \begin{bmatrix}H & A^{T}\\ A & 0 \end{bmatrix}\begin{bmatrix}x\\ y \end{bmatrix}=\begin{bmatrix}a\\ b \end{bmatrix},
/// \f]
/// where \eq{H} is the *Hessian matrix* of an objective function and \eq{A} is
/// the *Jacobian matrix* of a constraint function.
struct SaddlePointMatrix
{
    /// The Hessian matrix \eq{H} in the saddle point matrix.
    Vector H;

    /// The Jacobian matrix \eq{A} in the saddle point matrix.
    Matrix A;

    /// The indices of the fixed variables.
    /// The saddle point matrix has the following representation when some variables in \eq{x} are fixed:
    /// \f[
    /// \begin{bmatrix}H_{\mathrm{x}} & 0 & A_{\mathrm{x}}^{T}\\ 0 & I_{\mathrm{f}} & 0\\ A_{\mathrm{x}} & A_{\mathrm{f}} & 0 \end{bmatrix}\begin{bmatrix}x_{\mathrm{x}}\\ x_{\mathrm{f}}\\ y \end{bmatrix}=\begin{bmatrix}a_{\mathrm{x}}\\ a_{\mathrm{f}}\\ b \end{bmatrix},
    /// \f]
    /// where the subscripts \eq{\mathrm{x}} and \eq{\mathrm{f}} correspond to free and fixed
    /// variables, respectively.
    Indices ifixed;

    /// Convert this SaddlePointMatrix instance into a Matrix instance.
    auto matrix() const -> Matrix;
};

/// A type used to describe a saddle point right-hand side vector.
struct SaddlePointVector
{
    /// The saddle-point vector `x`.
    Vector x;

    /// The saddle-point vector `y`.
    Vector y;

    /// Convert this SaddlePointVector instance into a Vector instance.
    auto vector() const -> Vector;
};

/// Return the multiplication of a saddle point matrix and a saddle point vector.
auto operator*(const SaddlePointMatrix& A, const SaddlePointVector& x) -> SaddlePointVector;

} // namespace Optima

