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

namespace Optima {

// Forward declarations
class SaddlePointMatrix;
class SaddlePointVector;
class SaddlePointSolution;
class SaddlePointResult;

/// Used to describe the possible methods for solving saddle point problems.
enum class SaddlePointMethod
{
    PartialPivLU, FullPivLU, RangespaceDiagonal, Nullspace
};

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

    /// Set the saddle point method to a given one.
    auto setMethod(SaddlePointMethod method) -> void;

    /// Set the saddle point method to a *partial pivoting LU decomposition* algorithm.
    /// This method solves the saddle point problem by applying a partial pivoting
    /// LU decomposition to the saddle point matrix of dimension \eq{(n+m)\times(n+m)}.
    /// This method is in general accurate enough, but less accurate than its
    /// full pivoting counterpart. In general, it is also faster than the other methods
    /// for problems with small dimensions and when \eq{n} is not too larger than \eq{m}.
    /// @note This method takes no advantage of the particular structure of the saddle point matrix.
    auto setMethodPartialPivLU() -> void;

    /// Set the saddle point method to a *full pivoting LU decomposition* algorithm.
    /// This method solves the saddle point problems by applying a
    /// full pivoting LU decomposition to the saddle point matrix.
    /// It is in general very accurate, but also more computationally expensive.
    /// @note This method takes no advantage of the particular structure of the saddle point matrix.
    auto setMethodFullPivLU() -> void;

    /// Set the saddle point method to one based on a *rangespace* approach.
    /// This method reduces the saddle point problem of dimension \eq{n + m} to
    /// an equivalent one of dimension \eq{m}, where these dimensions are
    /// related to the dimensions of the Hessian matrix \eq{H}, \eq{n \times n},
    /// and Jacobian matrix \eq{A}, \eq{m \times n}.
    /// @warning This method should only be used when the Hessian matrix is a diagonal matrix.
    auto setMethodRangespaceDiagonal() -> void;

    /// Set the saddle point method to one based on a *nullspace* approach.
    /// This method reduces the saddle point problem of dimension \eq{n + m} to
    /// an equivalent one of dimension \eq{n - m}, where \eq{n \times n} is the
    /// dimension of the Hessian matrix \eq{H} and \eq{m \times n} is the
    /// dimension of the Jacobian matrix \eq{A}.
    auto setMethodNullspace() -> void;

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

    /// Return the current saddle point problem method.
    auto method() const -> SaddlePointMethod;

    /// Canonicalize the coefficient matrix \eq{A} of the saddle point problem.
    /// @note This method should be called before the @ref decompose method. However, it does not
    /// need to be called again if matrix \eq{A} of the saddle point problem is the same as in the
    /// last call to @ref canonicalize.
    /// @param lhs The coefficient matrix of the saddle point problem.
    auto canonicalize(const SaddlePointMatrix& lhs) -> SaddlePointResult;

    /// Decompose the coefficient matrix of the saddle point problem.
    /// @note This method should be called before the @ref solve method and after @ref canonicalize.
    /// @param lhs The coefficient matrix of the saddle point problem.
    auto decompose(const SaddlePointMatrix& lhs) -> SaddlePointResult;

    /// Solve the saddle point problem.
    /// @note This method expects that a call to method @ref decompose has already been performed.
    /// @param rhs The right-hand side vector of the saddle point problem.
    /// @param sol The solution of the saddle point problem.
    auto solve(const SaddlePointVector& rhs, SaddlePointSolution& sol) -> SaddlePointResult;

    /// Solve the saddle point problem.
    /// @note This method solves the saddle point problem without exploring
    /// @note the special structure of the saddle point matrix.
    /// @param lhs The left-hand side matrix of the saddle point problem.
    /// @param rhs The right-hand side vector of the saddle point problem.
    /// @param sol The solution of the saddle point problem.
    auto solve(const SaddlePointMatrix& lhs, const SaddlePointVector& rhs, SaddlePointSolution& sol) -> SaddlePointResult;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
