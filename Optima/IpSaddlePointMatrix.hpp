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
    MatrixXdConstRef H;

    /// The Jacobian matrix \eq{A} in the saddle point matrix.
    MatrixXdConstRef A;

    /// The diagonal matrix \eq{Z} in the saddle point matrix.
    VectorXdConstRef Z;

    /// The diagonal matrix \eq{W} in the saddle point matrix.
    VectorXdConstRef W;

    /// The diagonal matrix \eq{L} in the saddle point matrix.
    VectorXdConstRef L;

    /// The diagonal matrix \eq{U} in the saddle point matrix.
    VectorXdConstRef U;

    /// The number of free variables.
    Index nx;

    /// The number of fixed variables.
    Index nf;

    /// Construct a SaddlePointMatrix instance.
    /// @param H The \eq{H} matrix in the saddle point equation.
    /// @param A The \eq{A} matrix in the saddle point equation.
    /// @param Z The vector representing the diagonal matrix \eq{Z} in the saddle point equation.
    /// @param W The vector representing the diagonal matrix \eq{W} in the saddle point equation.
    /// @param L The vector representing the diagonal matrix \eq{L} in the saddle point equation.
    /// @param U The vector representing the diagonal matrix \eq{U} in the saddle point equation.
    /// @param nx The number of free variables.
    /// @param nf The number of fixed variables.
    IpSaddlePointMatrix(
        MatrixXdConstRef H,
        MatrixXdConstRef A,
        VectorXdConstRef Z,
        VectorXdConstRef W,
        VectorXdConstRef L,
        VectorXdConstRef U,
        Index nx, Index nf = 0);

    /// Convert this IpSaddlePointMatrix instance into a MatrixXd instance.
    operator MatrixXd() const;
};

/// A type used to describe an interior-point interior-point saddle point right-hand side vector.
class IpSaddlePointVector
{
public:
    /// The saddle-point solution vector \eq{a}.
    VectorXdConstRef a;

    /// The saddle-point solution vector \eq{b}.
    VectorXdConstRef b;

    /// The saddle-point solution vector \eq{c}.
    VectorXdConstRef c;

    /// The saddle-point solution vector \eq{d}.
    VectorXdConstRef d;

    /// Construct an IpSaddlePointVector instance with given \eq{(a,b,c,d)} vectors.
    /// @param a The right-hand side vector \eq{a} in the interior-point saddle point problem.
    /// @param b The right-hand side vector \eq{b} in the interior-point saddle point problem.
    /// @param c The right-hand side vector \eq{c} in the interior-point saddle point problem.
    /// @param d The right-hand side vector \eq{d} in the interior-point saddle point problem.
    IpSaddlePointVector(
        VectorXdConstRef a,
        VectorXdConstRef b,
        VectorXdConstRef c,
        VectorXdConstRef d);

    /// Construct an IpSaddlePointVector instance with given right-hand side vector.
    /// @param r The right-hand side vector \eq{r = (a, b, c, d)}.
    /// @param n The dimension of vectors \eq{a}, \eq{c}, \eq{d}.
    /// @param m The dimension of vector \eq{b}.
    IpSaddlePointVector(VectorXdConstRef r, Index n, Index m);

    /// Convert this IpSaddlePointVector instance into a VectorXd instance.
    operator VectorXd() const;
};

/// A type used to describe an interior-point saddle point solution vector.
class IpSaddlePointSolution
{
public:
    /// The solution vector \eq{x} in the interior-point saddle-point problem.
    VectorXdRef x;

    /// The solution vector \eq{y} in the interior-point saddle-point problem.
    VectorXdRef y;

    /// The solution vector \eq{z} in the interior-point saddle-point problem.
    VectorXdRef z;

    /// The solution vector \eq{w} in the interior-point saddle-point problem.
    VectorXdRef w;

    /// Construct an IpSaddlePointSolution instance with given \eq{(x,y,z,w)} vectors.
    /// @param x The solution vector \eq{x} in the interior-point saddle point problem.
    /// @param y The solution vector \eq{y} in the interior-point saddle point problem.
    /// @param w The solution vector \eq{z} in the interior-point saddle point problem.
    /// @param z The solution vector \eq{w} in the interior-point saddle point problem.
    IpSaddlePointSolution(
        VectorXdRef x,
        VectorXdRef y,
        VectorXdRef z,
        VectorXdRef w);

    /// Construct an IpSaddlePointSolution instance with given solution vector.
    /// @param s The solution vector \eq{s = (x, y, z, w)}.
    /// @param n The dimension of vectors \eq{x}, \eq{z}, \eq{w}.
    /// @param m The dimension of vector \eq{y}.
    IpSaddlePointSolution(VectorXdRef s, Index n, Index m);

    /// Assign this IpSaddlePointSolution instance with a VectorXdConstRef instance.
    auto operator=(VectorXdConstRef vec) -> IpSaddlePointSolution&;

    /// Convert this IpSaddlePointSolution instance into a VectorXd instance.
    operator VectorXd() const;
};

/// Return the multiplication of an IpSaddlePointMatrix by a vector.
auto operator*(IpSaddlePointMatrix lhs, VectorXdConstRef rhs) -> VectorXd;

} // namespace Optima

