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
#include <Optima/Matrix.hpp>
#include <Optima/VariantMatrix.hpp>

namespace Optima {

/// Used to represent the coefficient matrix in a saddle point problem of an interior-point algorithm.
/// An interior-point saddle point matrix is defined as a matrix with the following structure:
/// \f[
/// \begin{bmatrix}H & A^{T} & -I & -I\\A & 0 & 0 & 0\\Z & 0 & L & 0\\0 & W & 0 & U\end{bmatrix}\begin{bmatrix}x\\y\\z\\w\end{bmatrix}=\begin{bmatrix}a\\b\\c\\d\end{bmatrix}
/// \f]
/// where \eq{H} is the *Hessian matrix* of an objective function, \eq{A} is the *Jacobian matrix*
/// of a constraint function, and \eq{G} is a negative semi-definite matrix.
/// If the interior-point saddle point problem has fixed variables, then the saddle point matrix has the following representation:
/// \f[
/// \begin{bmatrix}H_{\mathrm{x}} & 0 & A_{\mathrm{x}}^{T} & -I_{\mathrm{x}} & 0 & -I_{\mathrm{x}} & 0\\0 & I_{\mathrm{f}} & 0 & 0 & 0 & 0 & 0\\A_{\mathrm{x}} & A_{\mathrm{f}} & 0 & 0 & 0 & 0 & 0\\Z_{\mathrm{x}} & 0 & 0 & L_{\mathrm{x}} & 0 & 0 & 0\\0 & 0 & 0 & 0 & I_{\mathrm{f}} & 0 & 0\\W_{\mathrm{x}} & 0 & 0 & 0 & 0 & U_{\mathrm{x}} & 0\\0 & 0 & 0 & 0 & 0 & 0 & I_{\mathrm{f}}\end{bmatrix}\begin{bmatrix}x_{\mathrm{x}}\\x_{\mathrm{f}}\\y\\z_{\mathrm{x}}\\z_{\mathrm{f}}\\w_{\mathrm{x}}\\w_{\mathrm{f}}\end{bmatrix}=\begin{bmatrix}a_{\mathrm{x}}\\a_{\mathrm{f}}\\b\\c_{\mathrm{x}}\\0\\d_{\mathrm{x}}\\0\end{bmatrix}
/// \f]
/// where the subscripts \eq{\mathrm{x}} and \eq{\mathrm{f}} correspond to free and fixed variables, respectively.
class IpSaddlePointMatrix
{
public:
    /// The Hessian matrix \eq{H} in the saddle point matrix.
    VariantMatrixConstRef H;

    /// The Jacobian matrix \eq{A} in the saddle point matrix.
    MatrixConstRef A;

    /// The diagonal matrix \eq{Z} in the saddle point matrix.
    VectorConstRef Z;

    /// The diagonal matrix \eq{W} in the saddle point matrix.
    VectorConstRef W;

    /// The diagonal matrix \eq{L} in the saddle point matrix.
    VectorConstRef L;

    /// The diagonal matrix \eq{U} in the saddle point matrix.
    VectorConstRef U;

    /// The indices of the fixed variables.
    VectorXiConstRef jf;

    /// Construct a SaddlePointMatrix instance.
    /// @param H The \eq{H} matrix in the saddle point equation.
    /// @param A The \eq{A} matrix in the saddle point equation.
    /// @param Z The vector representing the diagonal matrix \eq{Z} in the saddle point equation.
    /// @param W The vector representing the diagonal matrix \eq{W} in the saddle point equation.
    /// @param L The vector representing the diagonal matrix \eq{L} in the saddle point equation.
    /// @param U The vector representing the diagonal matrix \eq{U} in the saddle point equation.
    /// @param jf The indices of the fixed variables.
    IpSaddlePointMatrix(
        VariantMatrixConstRef H,
        MatrixConstRef A,
        VectorConstRef Z,
        VectorConstRef W,
        VectorConstRef L,
        VectorConstRef U,
        VectorXiConstRef jf);

    /// Convert this IpSaddlePointMatrix instance into a Matrix instance.
    operator Matrix() const;
};

/// A type used to describe an interior-point interior-point saddle point right-hand side vector.
class IpSaddlePointVector
{
public:
    /// The saddle point sub-vector \eq{x}.
    VectorConstRef x;

    /// The saddle point sub-vector \eq{y}.
    VectorConstRef y;

    /// The saddle point sub-vector \eq{z}.
    VectorConstRef z;

    /// The saddle point sub-vector \eq{w}.
    VectorConstRef w;

    /// The saddle point right-hand side sub-vector \eq{a} *(this is an alias to x)*.
    VectorConstRef a;

    /// The saddle point right-hand side sub-vector \eq{b} *(this is an alias to y)*.
    VectorConstRef b;

    /// The saddle point right-hand side sub-vector \eq{c} *(this is an alias to z)*.
    VectorConstRef c;

    /// The saddle point right-hand side sub-vector \eq{d} *(this is an alias to w)*.
    VectorConstRef d;

    /// Construct an IpSaddlePointVector instance with given \eq{(a,b,c,d)} vectors.
    /// @param x The saddle point sub-vector \eq{x}.
    /// @param y The saddle point sub-vector \eq{y}.
    /// @param z The saddle point sub-vector \eq{z}.
    /// @param w The saddle point sub-vector \eq{w}.
    IpSaddlePointVector(
        VectorConstRef x,
        VectorConstRef y,
        VectorConstRef z,
        VectorConstRef w);

    /// Construct an IpSaddlePointVector instance with given vector.
    /// @param r The vector \eq{r=\begin{bmatrix}x & y & z & w\end{bmatrix}}.
    /// @param n The dimension of vectors \eq{x}, \eq{z}, \eq{w}.
    /// @param m The dimension of vector \eq{y}.
    IpSaddlePointVector(VectorConstRef r, Index n, Index m);

    /// Convert this IpSaddlePointVector instance into a Vector instance.
    operator Vector() const;
};

/// A type used to describe an interior-point saddle point solution vector.
class IpSaddlePointSolution
{
public:
    /// The solution vector \eq{x} in the interior-point saddle point problem.
    VectorRef x;

    /// The solution vector \eq{y} in the interior-point saddle point problem.
    VectorRef y;

    /// The solution vector \eq{z} in the interior-point saddle point problem.
    VectorRef z;

    /// The solution vector \eq{w} in the interior-point saddle point problem.
    VectorRef w;

    /// Construct an IpSaddlePointSolution instance with given \eq{(x,y,z,w)} vectors.
    /// @param x The solution vector \eq{x} in the interior-point saddle point problem.
    /// @param y The solution vector \eq{y} in the interior-point saddle point problem.
    /// @param w The solution vector \eq{z} in the interior-point saddle point problem.
    /// @param z The solution vector \eq{w} in the interior-point saddle point problem.
    IpSaddlePointSolution(
        VectorRef x,
        VectorRef y,
        VectorRef z,
        VectorRef w);

    /// Construct an IpSaddlePointSolution instance with given solution vector.
    /// @param s The solution vector \eq{s = (x, y, z, w)}.
    /// @param n The dimension of vectors \eq{x}, \eq{z}, \eq{w}.
    /// @param m The dimension of vector \eq{y}.
    IpSaddlePointSolution(VectorRef s, Index n, Index m);

    /// Assign this IpSaddlePointSolution instance with a VectorConstRef instance.
    auto operator=(VectorConstRef vec) -> IpSaddlePointSolution&;

    /// Convert this IpSaddlePointSolution instance into a Vector instance.
    operator Vector() const;
};

/// Return the multiplication of an IpSaddlePointMatrix by a vector.
auto operator*(IpSaddlePointMatrix lhs, VectorConstRef rhs) -> Vector;

} // namespace Optima

