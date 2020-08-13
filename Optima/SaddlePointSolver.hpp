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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

// Forward declarations
class SaddlePointOptions;

/// The arguments for constructor of class SaddlePointSolver.
struct SaddlePointSolverInitArgs
{
    /// The dimension of vector *x* in the saddle point problem.
    Index nx = 0;

    /// The dimension of vector *p* in the saddle point problem.
    Index np = 0;

    /// The dimension of vector *y* in the saddle point problem (i.e. the number of rows in matrix *W = [Ax Ap; Jx Jp]*).
    Index m = 0;

    /// The upper and constant block *Ax* of matrix *W = [Ax Ap; Jx Jp]* in the saddle point problem.
    MatrixConstRef Ax;

    /// The upper and constant block *Ap* of matrix *W = [Ax Ap; Jx Jp]* in the saddle point problem.
    MatrixConstRef Ap;
};

/// The arguments for method SaddlePointSolver::canonicalize.
struct SaddlePointSolverCanonicalizeArgs
{
    /// The matrix block *Hxx* in *H = [Hxx Hxp; Hpx Hpp]* of the saddle point problem.
    MatrixConstRef Hxx;

    /// The matrix block *Hxp* in *H = [Hxx Hxp; Hpx Hpp]* of the saddle point problem.
    MatrixConstRef Hxp;

    /// The matrix block *Hpx* in *H = [Hxx Hxp; Hpx Hpp]* of the saddle point problem.
    MatrixConstRef Hpx;

    /// The matrix block *Hpp* in *H = [Hxx Hxp; Hpx Hpp]* of the saddle point problem.
    MatrixConstRef Hpp;

    /// The lower and variable block *Jx* of matrix *W = [Ax Ap; Jx Jp]* in the saddle point matrix.
    MatrixConstRef Jx;

    /// The lower and variable block *Jp* of matrix *W = [Ax Ap; Jx Jp]* in the saddle point matrix.
    MatrixConstRef Jp;

    /// The priority weights for variables in *x* to become basic variables.
    VectorConstRef wx;

    /// The indices of the variables *xu* in *x = (xs, xu)*.
    IndicesConstRef ju;
};

/// The arguments for method SaddlePointSolver::decompose.
struct SaddlePointSolverDecomposeArgs
{
    /// The matrix block *Hxx* in *H = [Hxx Hxp; Hpx Hpp]* of the saddle point problem.
    MatrixConstRef Hxx;

    /// The matrix block *Hxp* in *H = [Hxx Hxp; Hpx Hpp]* of the saddle point problem.
    MatrixConstRef Hxp;

    /// The matrix block *Hpx* in *H = [Hxx Hxp; Hpx Hpp]* of the saddle point problem.
    MatrixConstRef Hpx;

    /// The matrix block *Hpp* in *H = [Hxx Hxp; Hpx Hpp]* of the saddle point problem.
    MatrixConstRef Hpp;

    /// The lower and variable block *Jx* of matrix *W = [Ax Ap; Jx Jp]* in the saddle point matrix.
    MatrixConstRef Jx;

    /// The lower and variable block *Jp* of matrix *W = [Ax Ap; Jx Jp]* in the saddle point matrix.
    MatrixConstRef Jp;

    /// The indices of the variables *xu* in *x = (xs, xu)*.
    IndicesConstRef ju;
};

/// The arguments for method SaddlePointSolver::solve.
struct SaddlePointSolverSolveArgs
{
    /// The right-hand side vector *ax* in the saddle point problem.
    VectorConstRef ax;

    /// The right-hand side vector *ap* in the saddle point problem.
    VectorConstRef ap;

    /// The right-hand side vector *b = (bl, bn)* in the saddle point problem.
    VectorConstRef b;

    /// The solution vector *x* in the saddle point problem.
    VectorRef x;

    /// The solution vector *p* in the saddle point problem.
    VectorRef p;

    /// The solution vector *y = (yl, yn)* in the saddle point problem.
    VectorRef y;
};

/// The arguments for method SaddlePointSolver::solve.
struct SaddlePointSolverSolveAlternativeArgs
{
    /// The right-hand side vector *ax* (as input) and solution vector *x* (as output) in the saddle point problem.
    VectorRef x;

    /// The right-hand side vector *ap* (as input) and solution vector *p* (as output) in the saddle point problem.
    VectorRef p;

    /// The right-hand side vector *b* (as input) and solution vector *y* (as output) in the saddle point problem.
    VectorRef y;
};

/// The arguments for method SaddlePointSolver::solve.
/// When performing numerical optimization, the following saddle point problem
/// may emerge during a Newton step calculation:
///
/// @eqc{\begin{bmatrix}H & A^{T} & J^{T}\\ A & 0 & 0\\ J & 0 & 0
/// \end{bmatrix}\begin{bmatrix}\Delta x\\ \Delta y_{A}\\ \Delta y_{J}
/// \end{bmatrix}=-\begin{bmatrix}g+A^{T}y_{A}+J^{T}y_{J}\\ Ax-b\\ h
/// \end{bmatrix}}
///
/// Instead of dealing with delta variables, this can be formulated as follows:
///
/// @eqc{\begin{bmatrix}H & A^{T} & J^{T}\\ A & 0 & 0\\ J & 0 & 0
/// \end{bmatrix}\begin{bmatrix}\bar{x}\\ \bar{y}_{A}\\ \bar{y}_{J}
/// \end{bmatrix}=\begin{bmatrix}Hx-g\\ b\\ Jx-h \end{bmatrix}}
///
/// where @eq{x} is the current vector of primal variables, @eq{g} is
/// the current gradient vector of the objective function, @eq{h} is the
/// current residual of the non-linear constraint function @eq{h(x)}.
struct SaddlePointSolverSolveAdvancedArgs
{
    /// The right-hand side vector *x* in the saddle point problem.
    VectorConstRef x;

    /// The right-hand side vector *p* in the saddle point problem.
    VectorConstRef p;

    /// The right-hand side vector *fx* in the saddle point problem.
    VectorConstRef fx;

    /// The right-hand side vector *v* in the saddle point problem.
    VectorConstRef v;

    /// The right-hand side vector *b* in the saddle point problem.
    VectorConstRef b;

    /// The right-hand side vector *h* in the saddle point problem.
    VectorConstRef h;

    /// The solution vector *x* in the saddle point problem.
    VectorRef xbar;

    /// The solution vector *p* in the saddle point problem.
    VectorRef pbar;

    /// The solution vector *y* in the saddle point problem.
    VectorRef ybar;
};

/// The arguments for method SaddlePointSolver::residual.
struct SaddlePointSolverResidualArgs
{
    /// The vector *x* in the canonical residual equation.
    VectorConstRef x;

    /// The vector *p* in the canonical residual equation.
    VectorConstRef p;

    /// The right-hand side vector *b* in the canonical residual equation.
    VectorConstRef b;

    /// The output vector with the canonical residuals.
    VectorRef r;

    /// The output vector with the relative canonical residual errors.
    VectorRef e;
};

/// The arguments for method SaddlePointSolver::residual.
struct SaddlePointSolverResidualAdvancedArgs
{
    /// The vector *x* in the canonical residual equation.
    VectorConstRef x;

    /// The vector *p* in the canonical residual equation.
    VectorConstRef p;

    /// The right-hand side vector *b* in the canonical residual equation.
    VectorConstRef b;

    /// The right-hand side vector *h* in the canonical residual equation.
    VectorConstRef h;

    /// The output vector with the canonical residuals.
    VectorRef r;

    /// The output vector with the relative canonical residual errors.
    VectorRef e;
};

/// The arguments for method SaddlePointSolver::multiply.
struct SaddlePointSolverMultiplyArgs
{
    /// The vector *x* in the multiplication.
    VectorConstRef x;

    /// The vector *y* in the multiplication.
    VectorConstRef y;

    /// The result vector *a* in the multiplication.
    VectorRef a;

    /// The result vector *b* in the multiplication.
    VectorRef b;
};

/// The return type of method SaddlePointSolver::info.
struct SaddlePointSolverInfo
{
    /// The indices of the basic variables.
    IndicesConstRef jb;

    /// The indices of the non-basic variables.
    IndicesConstRef jn;

    /// The canonicalization matrix *R* of *W = [Ax Ap; Jx Jp]*.
    MatrixConstRef R;

    /// The matrix *S* in the canonical form of *W = [Ax Ap; Jx Jp]*.
    MatrixConstRef S;

    /// The permutation matrix *Q* in the canonical form of *W = [Ax Ap; Jx Jp]*.
    IndicesConstRef Q;
};

/// Used to solve saddle point problems.
/// Use this class to solve saddle point problems.
///
/// @note There is no need for matrix \eq{A} to have linearly independent rows.
/// The algorithm is able to ignore the linearly dependent rows automatically.
/// However, it is expected that vector \eq{b} in the saddle point matrix have
/// consistent values when linearly dependent rows in \eq{A} exists.
/// For example, assume \eq{Ax = b} represents:
/// \eqc{
/// \begin{bmatrix}1 & 1 & 1 & 1\\0 & 1 & 1 & 1\\1 & 0 & 0 & 0\end{bmatrix}\begin{bmatrix}x_{1}\\x_{2}\\x_{3}\\x_{4}\end{bmatrix}=\begin{bmatrix}b_{1}\\b_{2}\\b_{3}\end{bmatrix}.
/// }
/// Note that the third row of \eq{A} is linearly dependent on the other two
/// rows: \eq{\text{row}_3=\text{row}_1-\text{row}_2}.
/// Thus, it is expected that an input for vector \eq{b} is consistent with
/// the dependence relationship \eq{b_{3}=b_{1}-b_{2}}.
class SaddlePointSolver
{
public:
    /// Construct a SaddlePointSolver instance with given data.
    SaddlePointSolver(SaddlePointSolverInitArgs args);

    /// Construct a copy of a SaddlePointSolver instance.
    SaddlePointSolver(const SaddlePointSolver& other);

    /// Destroy this SaddlePointSolver instance.
    virtual ~SaddlePointSolver();

    /// Assign a SaddlePointSolver instance to this.
    auto operator=(SaddlePointSolver other) -> SaddlePointSolver&;

    /// Set the options for the solution of saddle point problems.
    auto setOptions(const SaddlePointOptions& options) -> void;

    /// Return the current saddle point options.
    auto options() const -> const SaddlePointOptions&;

    /// Canonicalize the *W = [Ax Ap; Jx Jp]* matrix of the saddle point problem.
    auto canonicalize(SaddlePointSolverCanonicalizeArgs args) -> void;

    /// Decompose the coefficient matrix of the saddle point problem into canonical form.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto decompose(SaddlePointSolverDecomposeArgs args) -> void;

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveArgs args) -> void;

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveAlternativeArgs args) -> void;

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolveAdvancedArgs args) -> void;

    /// Calculate the relative canonical residual of equation `W*x - b`.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto residuals(SaddlePointSolverResidualArgs args) -> void;

    /// Calculate the relative canonical residual of equation `W*x - [b; J*x + h]`.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto residuals(SaddlePointSolverResidualAdvancedArgs args) -> void;

    /// Calculate the multiplication of the saddle point matrix with a vector *(x, y)*.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto multiply(SaddlePointSolverMultiplyArgs args) -> void;

    /// Return the current state info of the saddle point solver.
    auto info() const -> SaddlePointSolverInfo;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
