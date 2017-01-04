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
#include <Optima/Common/Timing.hpp>
#include <Optima/Core/SaddlePointResult.hpp>

namespace Optima {

// Forward declarations
struct SaddlePointMatrix;
struct SaddlePointVector;

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
    auto solve(const SaddlePointVector& rhs, SaddlePointVector& sol) -> SaddlePointResult;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
