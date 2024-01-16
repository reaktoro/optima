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
#include <Optima/MasterVector.hpp>
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>

namespace Optima {

/// Used to backtrack a just performed step that produces out-of-bounds errors.
class BacktrackSearch
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a BacktrackSearch object.
    BacktrackSearch();

    /// Construct a copy of a BacktrackSearch object.
    BacktrackSearch(const BacktrackSearch& other);

    /// Destroy this BacktrackSearch object.
    virtual ~BacktrackSearch();

    /// Assign a BacktrackSearch object to this.
    auto operator=(BacktrackSearch other) -> BacktrackSearch&;

    /// Set the options for the backtrack search operation.
    auto setOptions(const BacktrackSearchOptions& options) -> void;

    /// Initialize this backtrack searcher once at the start of the optimization calculation.
    auto initialize(const MasterProblem& problem) -> void;

    /// Execute the backtrack search until `u` has no out-of-bounds and the new error has decreased.
    auto execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void;
};

} // namespace Optima
