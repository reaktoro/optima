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
#include <Optima/SaddlePointTypes.hpp>

namespace Optima {

/// Solve saddle point problems using a *rangespace approach*.
class SaddlePointSolverRangespace
{
public:
    /// Construct a default SaddlePointSolverRangespace instance.
    SaddlePointSolverRangespace(Index n, Index m);

    /// Construct a copy of a SaddlePointSolverRangespace instance.
    SaddlePointSolverRangespace(const SaddlePointSolverRangespace& other);

    /// Destroy this SaddlePointSolverRangespace instance.
    virtual ~SaddlePointSolverRangespace();

    /// Assign a SaddlePointSolverRangespace instance to this.
    auto operator=(SaddlePointSolverRangespace other) -> SaddlePointSolverRangespace&;

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void;

    /// Solve the canonical saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(CanonicalSaddlePointProblem args) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima