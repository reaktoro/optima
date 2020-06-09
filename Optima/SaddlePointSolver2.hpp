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

/// The arguments for constructor of class SaddlePointSolver2.
struct SaddlePointSolver2InitArgs
{
    /// The dimension of vector *x* in the saddle point problem (i.e. the number of columns in matrix *W = [A; J]*).
    Index n = 0;

    /// The dimension of vector *y* in the saddle point problem (i.e. the number of rows in matrix *W = [A; J]*).
    Index m = 0;

    /// The upper and constant block of the Jacobian matrix *W = [A; J]* in the saddle point problem.
    MatrixConstRef A;
};

/// The arguments for method SaddlePointSolver2::decompose.
struct SaddlePointSolver2DecomposeArgs
{
    /// The Hessian matrix *H* in the saddle point problem.
    MatrixConstRef H;

    /// The lower and variable block of the Jacobian matrix *W = [A; J]* in the saddle point problem.
    MatrixConstRef J;

    /// The negative semi-definite matrix *G* in the saddle point problem.
    MatrixConstRef G;

    /// The scaling diagonal matrix *D* in the saddle point problem.
    VectorConstRef D;

    /// The scaling diagonal matrix *V* in the saddle point problem.
    VectorConstRef V;

    /// The indices of the fixed variables.
    IndicesConstRef ifixed;

    /// The indices of rows in *H* whose off-diagonal entries are zero.
    IndicesConstRef idiagonal;
};

/// The arguments for method SaddlePointSolver2::solve.
struct SaddlePointSolver2SolveArgs
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

/// The arguments for method SaddlePointSolver2::solve.
struct SaddlePointSolver2SolveAlternativeArgs
{
    /// The right-hand side vector *a* (as input) and solution vector *x* (as output) in the saddle point problem.
    VectorRef x;

    /// The right-hand side vector *b* (as input) and solution vector *y* (as output) in the saddle point problem.
    VectorRef y;
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
class SaddlePointSolver2
{
public:
    /// Construct a default SaddlePointSolver2 instance.
    SaddlePointSolver2();

    /// Construct a SaddlePointSolver2 instance with given data.
    SaddlePointSolver2(SaddlePointSolver2InitArgs args);

    /// Construct a copy of a SaddlePointSolver2 instance.
    SaddlePointSolver2(const SaddlePointSolver2& other);

    /// Destroy this SaddlePointSolver2 instance.
    virtual ~SaddlePointSolver2();

    /// Assign a SaddlePointSolver2 instance to this.
    auto operator=(SaddlePointSolver2 other) -> SaddlePointSolver2&;

    /// Set the options for the solution of saddle point problems.
    auto setOptions(const SaddlePointOptions& options) -> void;

    /// Return the current saddle point options.
    auto options() const -> const SaddlePointOptions&;

    /// Decompose the coefficient matrix of the saddle point problem.
    /// @note This method should be called before the @ref solve method and after @ref canonicalize.
    /// @param lhs The coefficient matrix of the saddle point problem.
    auto decompose(SaddlePointSolver2DecomposeArgs args) -> void;

    /// Solve the saddle point problem.
    /// @note Method SaddlePointSolver2::decompose needs to be called first.
    auto solve(SaddlePointSolver2SolveArgs args) -> void;

    /// Solve the saddle point problem.
    /// @note Method SaddlePointSolver2::decompose needs to be called first.
    auto solve(SaddlePointSolver2SolveAlternativeArgs args) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
