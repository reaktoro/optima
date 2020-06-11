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
    /// The dimension of vector *x* in the saddle point problem (i.e. the number of columns in matrix *W = [A; J]*).
    Index n = 0;

    /// The dimension of vector *y* in the saddle point problem (i.e. the number of rows in matrix *W = [A; J]*).
    Index m = 0;

    /// The upper and constant block of the Jacobian matrix *W = [A; J]* in the saddle point problem.
    MatrixConstRef A;
};

/// The arguments for method SaddlePointSolver::decompose.
struct SaddlePointSolverDecomposeArgs
{
    /// The Hessian matrix *H* in the saddle point problem.
    MatrixConstRef H;

    /// The lower and variable block of the Jacobian matrix *W = [A; J]* in the saddle point problem.
    MatrixConstRef J;

    /// The negative semi-definite matrix *G* in the saddle point problem.
    MatrixConstRef G;

    /// The indices of the fixed variables.
    IndicesConstRef ifixed;
};

/// The arguments for method SaddlePointSolver::solve.
struct SaddlePointSolverSolveArgs
{
    /// The right-hand side vector *a* in the saddle point problem.
    VectorConstRef a;

    /// The right-hand side vector *b* in the saddle point problem.
    VectorConstRef b;

    /// The solution vector *x* in the saddle point problem.
    VectorRef x;

    /// The solution vector *y* in the saddle point problem.
    VectorRef y;
};

/// The arguments for method SaddlePointSolver::solve.
struct SaddlePointSolverSolveAlternativeArgs
{
    /// The right-hand side vector *a* (as input) and solution vector *x* (as output) in the saddle point problem.
    VectorRef x;

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
    /// The Hessian matrix *H* in the saddle point problem.
    MatrixConstRef H;

    /// The lower and variable block of the Jacobian matrix *W = [A; J]* in the saddle point problem.
    MatrixConstRef J;

    /// The right-hand side vector *x0* in the saddle point problem.
    VectorConstRef x;

    /// The right-hand side vector *g* in the saddle point problem.
    VectorConstRef g;

    /// The right-hand side vector *b* in the saddle point problem.
    VectorConstRef b;

    /// The right-hand side vector *h* in the saddle point problem.
    VectorConstRef h;

    /// The solution vector *x* in the saddle point problem.
    VectorRef xbar;

    /// The solution vector *y* in the saddle point problem.
    VectorRef ybar;
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
    /// Construct a default SaddlePointSolver instance.
    SaddlePointSolver();

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

    /// Decompose the coefficient matrix of the saddle point problem into canonical form.
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

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
