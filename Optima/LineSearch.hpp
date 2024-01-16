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
#include <Optima/LineSearchOptions.hpp>
#include <Optima/MasterVector.hpp>
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>

namespace Optima {

//=================================================================================================
// NOTE
//
// The Newton steps in Optima are currently generated in a way that prevents
// line search operations in general. This is because some variables on the
// bounds may have been considered in the Newton step calculation (because
// stability analysis indicated that they could potentially further decrease
// the object function if detached from their bounds). However, it may happen
// that once we compute the Newton step, the step for a variable on a lower
// bound is negative, or the step for a variable on the upper bound is
// positive. This indicates that these variables should then indeed remain on
// their bounds (at least for this iteration). Thus, the computed Newton step
// vector must be zeroed out for these variables on the bounds, otherwise the
// Newton step would cause these bounds to be violated. When we do this
// alteration on the Newton step, we change its properties (e.g., conservative
// properties if the linear constraints are mass/mole balance constraints). In
// addition, the modified Newton step may no longer be a descent direction with
// respect to the square of the residual errors.
//
// Thus, proper line search operations will require a recomputation of Newton
// step assumming that all variables currently on the bounds should be ignored,
// regardless of what the stability analysis is telling.
//=================================================================================================

/// Used to perform a line search minimization operation.
class LineSearch
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a LineSearch object.
    LineSearch();

    /// Construct a copy of a LineSearch object.
    LineSearch(const LineSearch& other);

    /// Destroy this LineSearch object.
    virtual ~LineSearch();

    /// Assign a LineSearch object to this.
    auto operator=(LineSearch other) -> LineSearch&;

    /// Set the options of this LineSearch object.
    auto setOptions(const LineSearchOptions& options) -> void;

    /// Initialize this LineSearch object once at the start of the optimization calculation.
    auto initialize(const MasterProblem& problem) -> void;

    /// Execute the line-search operation to decrease the current error using a minimization along the Newton direction.
    auto execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void;
};

} // namespace Optima
