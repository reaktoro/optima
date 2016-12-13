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
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointProblem.hpp>

namespace Optima {

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

    /// Set `true` to indicate that matrix `A` is a constant at every call to `solve`.
    auto constantA(bool isconst) -> void;

    /// Solve a saddle point problem.
    /// @param problem The saddle point problem.
    /// @param[in,out] solution The solution of the saddle point problem.
    auto solve(const SaddlePointProblem& problem, SaddlePointVector& solution) -> void;

    /// Solve a saddle point problem.
    /// @param problem The saddle point problem in canonical form.
    /// @param[in,out] solution The solution of the saddle point problem in canonical form.
    auto solve(const SaddlePointProblemCanonical& problem, SaddlePointVectorCanonical& solution) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

auto solve(SaddlePointProblemCanonical& problem, SaddlePointVectorCanonical& solution) -> void;

} // namespace Optima
