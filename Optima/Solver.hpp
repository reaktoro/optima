// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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

namespace Optima {

// Forward declarations
class Options;
class Problem;
class Result;
class Sensitivity;
class State;
struct Dims;

/// The solver for optimization problems.
class Solver
{
public:
    /// Construct a default Solver instance.
    Solver();

    /// Construct a copy of a Solver instance.
    Solver(const Solver& other);

    /// Destroy this Solver instance.
    virtual ~Solver();

    /// Assign a Solver instance to this.
    auto operator=(Solver other) -> Solver&;

    /// Set the options for the optimization calculation.
    auto setOptions(const Options& options) -> void;

    /// Solve the optimization problem.
    auto solve(const Problem& problem, State& state) -> Result;

    /// Solve the optimization problem and compute the sensitivity derivatives at the end.
    auto solve(const Problem& problem, State& state, Sensitivity& sensitivity) -> Result;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
