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

namespace Optima {

/// The options for the line search minimization operation.
struct LineSearchOptions
{
    /// The tolerance in the minimization calculation during the line search operation.
    double tolerance = 1.0e-5;

    /// The maximum number of iterations during the minimization calculation in the line search operation.
    unsigned maxiterations = 20;

    /// The parameter that triggers line-search when current error is greater than initial error by a given factor (`Enew > factor*E0`).
    double trigger_when_current_error_is_greater_than_initial_error_by_factor = 1.0;

    /// The parameter that triggers line-search when current error is greater than previous error by a given factor (`Enew > factor*Eold`).
    double trigger_when_current_error_is_greater_than_previous_error_by_factor = 2.0;
};

} // namespace Optima
