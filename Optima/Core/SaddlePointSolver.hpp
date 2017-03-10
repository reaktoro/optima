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
#include <memory>

// Optima includes
#include <Optima/Common/Index.hpp>
#include <Optima/Math/Matrix.hpp>

namespace Optima {

// Forward declarations
class  SaddlePointMatrix;
struct SaddlePointOptions;
class  SaddlePointResult;
class  SaddlePointSolution;
class  SaddlePointVector;

/// Used to solve saddle point problems.
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

    /// Set the saddle point method to the one that finishes in less time.
    /// This method solves the same saddle point problem many times for each
    /// supported saddle point method. It then averages the time spent for each method
    /// and chooses the one that finished faster on average.
    /// @warning This method should be used with care, possibly once during
    /// @warning an initialization stage.
    /// @param n The number of columns of the Jacobian matrix \eq{A} and Hessian matrix \eq{H}.
    /// @param m The number of rows of the Jacobian matrix \eq{A}.
    auto setMethodMoreEfficient(Index n, Index m) -> void;

    /// Set the saddle point method to the one that produces less numerical error.
    /// The numerical error is calculated as:
    /// \eqc{e = ||\mathrm{lhs} \times \mathrm{sol} - \mathrm{rhs}||/||\mathrm{rhs}||.}
    /// @param lhs The saddle point matrix of the saddle point problem.
    /// @param rhs The saddle point right-hand side vector of the saddle point problem.
    auto setMethodMoreAccurate(const SaddlePointMatrix& lhs, const SaddlePointVector& rhs) -> void;

    /// Return the current saddle point options.
    auto options() const -> const SaddlePointOptions&;

    /// Canonicalize the coefficient matrix \eq{A} of the saddle point problem.
    /// @note This method should be called before the @ref decompose method. However, it does not
    /// need to be called again if matrix \eq{A} of the saddle point problem is the same as in the
    /// last call to @ref canonicalize.
    /// @param A The coefficient matrix \eq{A} of the saddle point problem.
    auto canonicalize(MatrixXdConstRef A) -> SaddlePointResult;

    /// Decompose the coefficient matrix of the saddle point problem.
    /// @note This method should be called before the @ref solve method and after @ref canonicalize.
    /// @param lhs The coefficient matrix of the saddle point problem.
    auto decompose(SaddlePointMatrix lhs) -> SaddlePointResult;

    /// Solve the saddle point problem.
    /// @note This method expects that a call to method @ref decompose has already been performed.
    /// @param rhs The right-hand side vector of the saddle point problem.
    /// @param sol The solution of the saddle point problem.
    auto solve(SaddlePointVector rhs, SaddlePointSolution sol) -> SaddlePointResult;

    /// Solve the saddle point problem.
    /// @note This method solves the saddle point problem without exploring
    /// @note the special structure of the saddle point matrix.
    /// @param lhs The left-hand side matrix of the saddle point problem.
    /// @param rhs The right-hand side vector of the saddle point problem.
    /// @param sol The solution of the saddle point problem.
    auto solve(SaddlePointMatrix lhs, SaddlePointVector rhs, SaddlePointSolution sol) -> SaddlePointResult;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
