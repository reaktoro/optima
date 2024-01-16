// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2023 Allan Leal
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

// Optima includes
#include <Optima/ConvergenceCheck.hpp>

namespace Optima {

/// Used to organize the options for convergence analysis.
struct ConvergenceOptions
{
    /// The tolerance for the optimality error.
    double tolerance = 1.0e-8;

    /// An optional convergence check function to be used in addition to default check.
    std::function<bool(ConvergenceCheckArgs const&)> check;
};

} // namespace Optima
