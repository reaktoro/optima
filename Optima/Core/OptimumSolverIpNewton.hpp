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

namespace Optima {

// Forward declarations
struct OptimumOptions;
struct OptimumProblem;
struct OptimumResult;
struct OptimumState;

/// The class that implements the IpNewton algorithm using an interior-point method.
class OptimumSolverIpNewton
{
public:
    /// Construct a default OptimumSolverIpNewton instance.
    OptimumSolverIpNewton();

    /// Construct a copy of an OptimumSolverIpNewton instance.
    OptimumSolverIpNewton(const OptimumSolverIpNewton& other);

    /// Destroy this OptimumSolverIpNewton instance.
    virtual ~OptimumSolverIpNewton();

    /// Assign an OptimumSolverIpNewton instance to this.
    auto operator=(OptimumSolverIpNewton other) -> OptimumSolverIpNewton&;

    /// Set the options for the optimization calculation.
    auto setOptions(const OptimumOptions& options) -> void;

    /// Initialize the optimization solver with the structure of the problem.
    auto initialize(const OptimumStructure& structure) -> void;

    /// Solve an optimization problem.
    /// This method is useful when the same optimization problem needs to
    /// be solved multiple times, but with only different parameters.
    /// @note This method expects that the structure of the
    /// @note optimization problem was set with method @ref initialize.
    /// @param params The parameters for the optimization calculation.
    /// @param state[in,out] The initial guess and the final state of the optimization calculation.
    auto solve(const OptimumParams& params, OptimumState& state) -> OptimumResult;

    /// Solve an optimization problem.
    /// @param problem The definition of the optimization problem.
    /// @param state[in,out] The initial guess and the final state of the optimization calculation.
    auto solve(const OptimumProblem& problem, OptimumState& state) -> OptimumResult;

    /// Return the sensitivity `dx/dp` of the solution `x` with respect to a vector of parameters `p`.
    /// @param dgdp The derivatives `dg/dp` of the objective gradient `grad(f)` with respect to the parameters `p`
    /// @param dbdp The derivatives `db/dp` of the vector `b` with respect to the parameters `p`
    auto dxdp(ConstVectorRef dgdp, ConstVectorRef dbdp) -> VectorXd;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
