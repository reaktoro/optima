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
#include <Optima/NewtonStepOptions.hpp>

namespace Optima {

// Forward declarations
class JacobianMatrix;
class ResidualVector;
class SolutionVector;

/// Used to calculate Newton steps in the optimization calculation.
class NewtonStepSolver
{
public:
    /// Construct a NewtonStepSolver instance.
    NewtonStepSolver(Index nx, Index np, Index ny, Index nz);

    /// Construct a copy of a NewtonStepSolver instance.
    NewtonStepSolver(const NewtonStepSolver& other);

    /// Destroy this NewtonStepSolver instance.
    virtual ~NewtonStepSolver();

    /// Assign a NewtonStepSolver instance to this.
    auto operator=(NewtonStepSolver other) -> NewtonStepSolver&;

    /// Set the options for the Newton step calculation.
    auto setOptions(const NewtonStepOptions& options) -> void;

    /// Return the current options of this Newton step solver.
    auto options() const -> const NewtonStepOptions&;

    /// Decompose the Jacobian matrix.
    auto decompose(const JacobianMatrix& J) -> void;

    /// Compute the Newton step.
    /// @warning Ensure method @ref decompose has already been called.
    /// This will allow you to reuse the decomposition of the Jacobian
    /// matrix for multiple Newton step computations if needed.
    /// @param J The Jacobian matrix in the Newton step matrix problem.
    /// @param F The residual vector in the Newton step matrix problem.
    /// @param[out] du The calculated Newton step *du = (dx, dp, dy, dz)*.
    auto compute(const JacobianMatrix& J, const ResidualVector& F, SolutionVector& du) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
