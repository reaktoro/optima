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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

// Forward declarations
class IpSaddlePointMatrix;
class IpSaddlePointSolution;
class IpSaddlePointVector;
class SaddlePointOptions;
class Result;

/// Used to solve saddle point problems in interior-point algorithms.
class IpSaddlePointSolver
{
public:
    /// Construct a default IpSaddlePointSolver instance.
    IpSaddlePointSolver();

    /// Construct a copy of a IpSaddlePointSolver instance.
    IpSaddlePointSolver(const IpSaddlePointSolver& other);

    /// Destroy this IpSaddlePointSolver instance.
    virtual ~IpSaddlePointSolver();

    /// Assign a IpSaddlePointSolver instance to this.
    auto operator=(IpSaddlePointSolver other) -> IpSaddlePointSolver&;

    /// Set the options for the solution of saddle point problems.
    auto setOptions(const SaddlePointOptions& options) -> void;

    /// Return the current saddle point options.
    auto options() const -> const SaddlePointOptions&;

    /// Initialize the saddle point solver with the coefficient matrix \eq{A} of the saddle point problem.
    /// @note This method should be called before the @ref decompose method. However, it does not
    /// need to be called again if matrix \eq{A} of the saddle point problem is the same as in the
    /// last call to @ref initialize.
    /// @param A The coefficient matrix \eq{A} of the saddle point problem.
    auto initialize(MatrixConstRef A) -> Result;

    /// Decompose the coefficient matrix of the saddle point problem.
    /// @note This method should be called before the @ref solve method and after @ref canonicalize.
    /// @param lhs The coefficient matrix of the saddle point problem.
    auto decompose(IpSaddlePointMatrix lhs) -> Result;

    /// Solve the saddle point problem.
    /// @note This method expects that a call to method @ref decompose has already been performed.
    /// @param lhs The coefficient matrix of the saddle point problem.
    /// @param rhs The right-hand side vector of the saddle point problem.
    /// @param sol The solution of the saddle point problem.
    auto solve(IpSaddlePointVector rhs, IpSaddlePointSolution sol) -> Result;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
