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
class SaddlePointMatrix;
class SaddlePointOptions;
class SaddlePointSolution;
class SaddlePointVector;

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

    /// Initialize the saddle point solver with the coefficient matrix \eq{A} of the saddle point problem.
    /// @note This method should be called before the @ref decompose method. However, it does not
    /// need to be called again if matrix \eq{A} of the saddle point problem is the same as in the
    /// last call to @ref initialize.
    /// @param A The coefficient matrix \eq{A} of the saddle point problem.
    auto initialize(MatrixConstRef A) -> void;

    /// Decompose the coefficient matrix of the saddle point problem.
    /// @note This method should be called before the @ref solve method and after @ref canonicalize.
    /// @param lhs The coefficient matrix of the saddle point problem.
    auto decompose(SaddlePointMatrix lhs) -> void;

    /// Solve the saddle point problem.
    /// @note This method expects that a call to method @ref decompose has already been performed.
    /// @param lhs The coefficient matrix of the saddle point problem.
    /// @param rhs The right-hand side vector of the saddle point problem.
    /// @param sol The solution of the saddle point problem.
    auto solve(SaddlePointVector rhs, SaddlePointSolution sol) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
