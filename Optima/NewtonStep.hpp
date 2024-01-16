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
#include <Optima/MasterVector.hpp>
#include <Optima/NewtonStepOptions.hpp>
#include <Optima/ResidualFunction.hpp>

namespace Optima {

/// Used to update the variables in an optimization problem using Newton steps.
class NewtonStep
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a NewtonStep object.
    NewtonStep();

    /// Construct a copy of a NewtonStep object.
    NewtonStep(const NewtonStep& other);

    /// Destroy this NewtonStep object.
    virtual ~NewtonStep();

    /// Assign a NewtonStep object to this.
    auto operator=(NewtonStep other) -> NewtonStep&;

    /// Set the options of this NewtonStep object.
    auto setOptions(const NewtonStepOptions& options) -> void;

    /// Initialize this NewtonStep object once at the start of the optimization calculation.
    auto initialize(const MasterProblem& problem) -> void;

    /// Apply Newton step to compute the next state of master variables.
    auto apply(const ResidualFunction& F, MasterVectorView uo, MasterVectorRef u) -> void;
};

} // namespace Optima
