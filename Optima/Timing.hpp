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
#include <chrono>

namespace Optima {

/// A type alias for time related numbers.
using Time = std::chrono::time_point<std::chrono::high_resolution_clock>;

/// Used to measure time between two execution points.
class Timer
{
public:
	/// Construct a default Timer instance.
	Timer();

	/// Return the elapsed time since object creation (in seconds)
	auto elapsed() const -> double;

	/// Convert this Time instance into double (time in seconds).
	operator double() const;

private:
	/// The time at which this Time object was created.
	Time begin;
};

using Duration = std::chrono::duration<double>;

/// Return the time point now
/// @see elapsed
auto timenow() -> Time;

/// Return the elapsed time between two time points (in unit of s)
/// @param end The end time point
/// @param end The begin time point
/// @return The elapsed time between *end* and *begin* in seconds
auto elapsed(const Time& end, const Time& begin) -> double;

/// Return the elapsed time between a time point and now (in unit of s)
/// @param end The begin time point
/// @return The elapsed time between now and *begin* in seconds
auto elapsed(const Time& begin) -> double;

} // namespace Optima
