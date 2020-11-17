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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/MasterVector.hpp>
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>

namespace Optima {

/// Used to reduce the error level of last performed step if needed.
class ErrorControl
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a ErrorControl object.
    ErrorControl(const MasterDims& dims);

    /// Construct a copy of a ErrorControl object.
    ErrorControl(const ErrorControl& other);

    /// Destroy this ErrorControl object.
    virtual ~ErrorControl();

    /// Assign a ErrorControl object to this.
    auto operator=(ErrorControl other) -> ErrorControl&;

    /// Execute the error control operation to potentially decrease error level.
    auto execute(const MasterProblem& problem, MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void;
};

} // namespace Optima
