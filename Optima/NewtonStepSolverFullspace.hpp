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
#include <Optima/CanonicalVector.hpp>
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

// Forward declarations
class JacobianMatrix;
class ResidualVector;

/// Used to calculate Newton steps in the optimization calculation.
class NewtonStepSolverFullspace
{
public:
    /// Construct a NewtonStepSolverFullspace instance.
    NewtonStepSolverFullspace(Index nx, Index np, Index ny, Index nz);

    /// Construct a copy of a NewtonStepSolverFullspace instance.
    NewtonStepSolverFullspace(const NewtonStepSolverFullspace& other);

    /// Destroy this NewtonStepSolverFullspace instance.
    virtual ~NewtonStepSolverFullspace();

    /// Assign a NewtonStepSolverFullspace instance to this.
    auto operator=(NewtonStepSolverFullspace other) -> NewtonStepSolverFullspace&;

    /// Decompose the Jacobian matrix.
    auto decompose(const JacobianMatrix& M) -> void;

    /// Compute the Newton step.
    /// Using this method presumes method @ref decompose has already been
    /// called. This will allow you to reuse the decomposition of the Jacobian
    /// matrix for the multiple Newton step computations if needed.
    /// @param F The residual vector in the Newton step matrix problem.
    /// @param[out] dus The calculated canonical Newton step *dus = (dxs, dp, dwbs)*.
    auto compute(const JacobianMatrix& J, const ResidualVector& F, CanonicalVectorRef dus) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
