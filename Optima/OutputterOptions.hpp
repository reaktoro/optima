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
#include <string>

namespace Optima {

/// The type that describes the options for the output of an optimization calculation.
struct OutputterOptions
{
    /// The option that enable the output of the calculation.
    bool active = false;

    /// The option that indicates that the floating-point values should be in fixed notation.
    bool fixed = false;

    /// The option that indicates that the floating-point values should be in scientific notation.
    bool scientific = false;

    /// The precision of the floating-point values in the output.
    unsigned precision = 6;

    /// The width of the columns in the output.
    unsigned width = 15;

    /// The string used to separate the columns in the output.
    std::string separator = "|";

    /// The name of the file where the output will be written.
    std::string filename = "optima.log.txt";
};

} // namespace
