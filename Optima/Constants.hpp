// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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

constexpr auto SUCCEEDED = true;
constexpr auto FAILED    = false;

/// The floating-point representation of positive infinity
constexpr auto inf = std::numeric_limits<double>::infinity();

/// The floating-point representation of NaN
constexpr auto NaN = std::numeric_limits<double>::quiet_NaN();

} // namespace Optima
