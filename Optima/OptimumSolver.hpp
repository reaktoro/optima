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
#include <Optima/Matrix.hpp>
#include <Optima/Objective.hpp>

namespace Optima {

// Forward declarations
class OptimumOptions;
class OptimumParams;
class OptimumProblem;
class OptimumResult;
class OptimumState;
class OptimumStructure;

/// The class that implements the IpNewton algorithm using an interior-point method.
class OptimumSolver
{
public:
    /// Construct an OptimumSolver instance with given optimization structure.
    OptimumSolver(const OptimumStructure& structure);

    /// Construct a copy of an OptimumSolver instance.
    OptimumSolver(const OptimumSolver& other);

    /// Destroy this OptimumSolver instance.
    virtual ~OptimumSolver();

    /// Assign an OptimumSolver instance to this.
    auto operator=(OptimumSolver other) -> OptimumSolver&;

    /// Set the options for the optimization calculation.
    auto setOptions(const OptimumOptions& options) -> void;

    /// Solve an optimization problem.
    /// This method is useful when the same optimization problem needs to
    /// be solved multiple times, but with only different parameters.
    /// @note This method expects that the structure of the
    /// @note optimization problem was set with method @ref initialize.
    /// @param params The parameters for the optimization calculation.
    /// @param state[in,out] The initial guess and the final state of the optimization calculation.
    auto solve(const OptimumParams& params, OptimumState& state) -> OptimumResult;

    /// Return the sensitivity \eq{dx/dp} of the solution \eq{x} with respect to a vector of parameters \eq{p}.
    /// @param dgdp The derivatives \eq{dg/dp} of the gradient vector \eq{g = \nabla f} with respect to the parameters \eq{p}.
    /// @param dbdp The derivatives \eq{db/dp} of the vector \eq{b} with respect to the parameters \eq{p}.
    auto dxdp(VectorConstRef dgdp, VectorConstRef dbdp) -> Vector;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
