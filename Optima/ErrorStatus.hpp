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
#include <Optima/ErrorStatusOptions.hpp>
#include <Optima/ResidualErrors.hpp>

namespace Optima {

/// Used to manage the current error status duing an optimization calculation.
class ErrorStatus
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a ErrorStatus object.
    ErrorStatus();

    /// Construct a copy of a ErrorStatus object.
    ErrorStatus(const ErrorStatus& other);

    /// Destroy this ErrorStatus object.
    virtual ~ErrorStatus();

    /// Assign a ErrorStatus object to this.
    auto operator=(ErrorStatus other) -> ErrorStatus&;

    /// Set the options for the error status checking.
    auto setOptions(const ErrorStatusOptions& options) -> void;

    /// Initialize this error status checker.
    auto initialize() -> void;

    /// Update the error status.
    auto update(const ResidualErrors& E) -> void;

    /// Return `true` if current error has decreased since last update.
    auto errorHasDecreased() const -> bool;

    /// Return `true` if current error has decreased significantly since last update.
    auto errorHasDecreasedSignificantly() const -> bool;

    /// Return `true` if current error has increased since last update.
    auto errorHasIncreased() const -> bool;

    /// Return `true` if current error has increased significantly since last update.
    auto errorHasIncreasedSignificantly() const -> bool;

    /// Return `true` if current error is not a finite number (e.g., `inf` or `nan`).
    auto errorIsntFinite() const -> bool;

    /// Return the current error in the optimization calculation.
    auto error() const -> double;
};

} // namespace Optima
