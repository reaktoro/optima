// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
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
#include <Optima/Timing.hpp>

namespace Optima {

/// Used to indicate the result of the evaluation of a function.
class Result
{
public:
    /// Construct a default Result instance.
    Result();

    /// Set the success of the calculation.
    auto success(bool value) -> void;

    /// Return `true` if the calculation was successful.
    auto success() const -> bool;

    /// Return the elapsed time in seconds of the calculation.
    auto time() const -> double;

    /// Start the stopwatch.
    auto start() -> Result&;

    /// Stop the stopwatch.
    auto stop() -> Result&;

    /// Accumulate the result of several saddle point problem operations.
    auto operator+=(const Result& other) -> Result&;

    /// Accumulate the result of several saddle point problem operations.
    auto operator+(const Result& other) const -> Result;

private:
    /// True if the calculation was successful.
    bool m_success;

    /// The elapsed time in seconds of the calculation.
    double m_time;

    /// The time at which start method was called.
    Time m_start;

    /// The time at which stop method was called.
    Time m_stop;
};

} // namespace Optima
