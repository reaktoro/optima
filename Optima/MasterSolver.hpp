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
#include <Optima/MasterProblem.hpp>
#include <Optima/MasterSensitivity.hpp>
#include <Optima/MasterState.hpp>
#include <Optima/MasterVector.hpp>
#include <Optima/Options.hpp>
#include <Optima/Result.hpp>

namespace Optima {

/// Used for solving master optimization problems.
class MasterSolver
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a default MasterSolver object.
    MasterSolver();

    /// Construct a copy of a MasterSolver object.
    MasterSolver(const MasterSolver& other);

    /// Destroy this MasterSolver object.
    virtual ~MasterSolver();

    /// Assign a MasterSolver object to this.
    auto operator=(MasterSolver other) -> MasterSolver&;

    /// Set the options for the master optimization calculation.
    auto setOptions(const Options& options) -> void;

    /// Solve the given master optimization problem.
    auto solve(const MasterProblem& problem, MasterState& state) -> Result;

    /// Solve the given master optimization problem and compute the sensitivity derivatives at the end.
    auto solve(const MasterProblem& problem, MasterState& state, MasterSensitivity& sensitivity) -> Result;
};

} // namespace Optima
