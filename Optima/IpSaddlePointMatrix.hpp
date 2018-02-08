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
#include <Optima/Math/Matrix.hpp>

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

    /// Return the Hessian matrix \eq{H} in the saddle point matrix.
    auto H() const -> MatrixXdConstRef { return m_H; }

    /// Return the Jacobian matrix \eq{A} in the saddle point matrix.
    auto A() const -> MatrixXdConstRef { return m_A; }

    /// Return the diagonal matrix \eq{Z} in the saddle point matrix.
    auto Z() const -> VectorXdConstRef { return m_Z; }

    /// Return the diagonal matrix \eq{W} in the saddle point matrix.
    auto W() const -> VectorXdConstRef { return m_W; }

    /// Return the diagonal matrix \eq{L} in the saddle point matrix.
    auto L() const -> VectorXdConstRef { return m_L; }

    /// Return the diagonal matrix \eq{U} in the saddle point matrix.
    auto U() const -> VectorXdConstRef { return m_U; }

    /// Return the number of free variables.
    auto nx() const -> Index { return m_nx; }

    /// Return the number of fixed variables.
    auto nf() const -> Index { return m_nf; }

    /// Return the number of variables.
    auto n() const -> Index { return m_nx + m_nf; }

    /// Return the number of linear equality constraints.
    auto m() const -> Index { return m_A.rows(); }

    /// Return the dimension of the saddle point matrix.
    auto size() const -> Index { return 3*n() + m(); }

    /// Convert this IpSaddlePointMatrix instance into a Matrix instance.
    auto matrix() const -> MatrixXd;

private:
    /// The Hessian matrix \eq{H} in the saddle point matrix.
    MatrixXdConstRef m_H;

    /// The Jacobian matrix \eq{A} in the saddle point matrix.
    MatrixXdConstRef m_A;

    /// The diagonal matrix \eq{Z} in the saddle point matrix.
    VectorXdConstRef m_Z;

    /// The diagonal matrix \eq{W} in the saddle point matrix.
    VectorXdConstRef m_W;

    /// The diagonal matrix \eq{L} in the saddle point matrix.
    VectorXdConstRef m_L;

    /// The diagonal matrix \eq{U} in the saddle point matrix.
    VectorXdConstRef m_U;

    /// The number of free variables.
    Index m_nx;

    /// The number of fixed variables.
    Index m_nf;
};

/// A type used to describe an interior-point interior-point saddle point right-hand side vector.
class IpSaddlePointVector
{
public:
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

    /// Return the dimension of the interior-point saddle point vector.
    auto size() const -> Index { return m_a.rows() + m_b.rows() + m_c.rows() + m_d.rows(); }

    /// Return the solution vector \eq{a}.
    auto a() const -> VectorXdConstRef { return m_a; }

    /// Return the solution vector \eq{b}.
    auto b() const -> VectorXdConstRef { return m_b; }

    /// Return the solution vector \eq{c}.
    auto c() const -> VectorXdConstRef { return m_c; }

    /// Return the solution vector \eq{d}.
    auto d() const -> VectorXdConstRef { return m_d; }

    /// Convert this IpSaddlePointVector instance into a Vector instance.
    auto vector() const -> VectorXd;

private:
    /// The saddle-point solution vector \eq{a}.
    VectorXdConstRef m_a;

    /// The saddle-point solution vector \eq{b}.
    VectorXdConstRef m_b;

    /// The saddle-point solution vector \eq{c}.
    VectorXdConstRef m_c;

    /// The saddle-point solution vector \eq{d}.
    VectorXdConstRef m_d;
};

/// A type used to describe an interior-point saddle point solution vector.
class IpSaddlePointSolution
{
public:
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

    /// Return the dimension of the interior-point saddle point solution vector.
    auto size() const -> Index { return m_x.rows() + m_y.rows() + m_z.rows() + m_w.rows(); }

    /// Return the solution vector \eq{x}.
    auto x() -> VectorXdRef { return m_x; }

    /// Return the solution vector \eq{y}.
    auto y() -> VectorXdRef { return m_y; }

    /// Return the solution vector \eq{z}.
    auto z() -> VectorXdRef { return m_z; }

    /// Return the solution vector \eq{w}.
    auto w() -> VectorXdRef { return m_w; }

    /// Convert this IpSaddlePointSolution instance into a Vector instance.
    auto vector() const -> VectorXd;

private:
    /// The solution vector \eq{x} in the interior-point saddle-point problem.
    VectorXdRef m_x;

    /// The solution vector \eq{y} in the interior-point saddle-point problem.
    VectorXdRef m_y;

    /// The solution vector \eq{z} in the interior-point saddle-point problem.
    VectorXdRef m_z;

    /// The solution vector \eq{w} in the interior-point saddle-point problem.
    VectorXdRef m_w;
};

/// Assign a MatrixXdRef instance with an IpSaddlePointMatrix instance.
auto operator<<(MatrixXdRef mat, const IpSaddlePointMatrix& lhs) -> MatrixXdRef;

/// Assign a VectorXdRef instance with an IpSaddlePointVector instance.
auto operator<<(VectorXdRef vec, const IpSaddlePointVector& rhs) -> VectorXdRef;

/// Return the multiplication of an IpSaddlePointMatrix by a vector.
auto operator*(const IpSaddlePointMatrix& lhs, VectorXdConstRef rhs) -> VectorXd;

} // namespace Optima

