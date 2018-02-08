// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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

// Optima includes
#include <Optima/Timing.hpp>

namespace Optima {

/// Used to indicate the result details of a saddle point problem calculation.
class SaddlePointResult
{
public:
    /// Construct a default SaddlePointResult instance.
    SaddlePointResult();

    /// Return `true` if the calculation was successful.
    auto success() const -> bool;

    /// Return the elapsed time in seconds of the calculation.
    auto time() const -> double;

    /// Start the stopwatch.
    auto start() -> SaddlePointResult&;

    /// Stop the stopwatch.
    auto stop() -> SaddlePointResult&;

    /// Stop the stopwatch and set success to false.
    /// @param errormsg The error message about the failure.
    auto failed(std::string error) -> SaddlePointResult&;

    /// Return the error message about the failure.
    auto error() -> std::string;

    /// Accumulate the result of several saddle point problem operations.
    auto operator+=(const SaddlePointResult& other) -> SaddlePointResult&;

    /// Accumulate the result of several saddle point problem operations.
    auto operator+(const SaddlePointResult& other) const -> SaddlePointResult;

private:
    /// True if the calculation was successful.
    bool m_success;

    /// The elapsed time in seconds of the calculation.
    double m_time;

    /// The time at which start method was called.
    Time m_start;

    /// The time at which stop method was called.
    Time m_stop;

    /// The error message
    std::string m_error;
};

} // namespace Optima
