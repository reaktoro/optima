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
#include <Optima/BacktrackSearchOptions.hpp>
#include <Optima/ErrorStatusOptions.hpp>
#include <Optima/LineSearchOptions.hpp>
#include <Optima/MasterVector.hpp>
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>

namespace Optima {

/// The options for error control in the optimization calculation.
struct ErrorControlOptions
{
    /// The options for assessing error status.
    ErrorStatusOptions errorstatus;

    /// The options for the backtrack search operation.
    BacktrackSearchOptions backtracksearch;

    /// The options for the linear search minimization operation.
    LineSearchOptions linesearch;
};

/// Used to reduce the error level of last performed step if needed.
class ErrorControl
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a ErrorControl object.
    ErrorControl();

    /// Construct a copy of a ErrorControl object.
    ErrorControl(const ErrorControl& other);

    /// Destroy this ErrorControl object.
    virtual ~ErrorControl();

    /// Assign a ErrorControl object to this.
    auto operator=(ErrorControl other) -> ErrorControl&;

    /// Set the options for the error control in the optimization calculation.
    auto setOptions(const ErrorControlOptions& options) -> void;

    /// Initialize this error control object once at the start of the optimization calculation.
    auto initialize(const MasterProblem& problem) -> void;

    /// Execute the error control operation to potentially decrease error level.
    auto execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void;
};

} // namespace Optima
