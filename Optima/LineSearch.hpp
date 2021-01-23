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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/LineSearchOptions.hpp>
#include <Optima/MasterVector.hpp>
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>

namespace Optima {

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

    /// Start the backtrack search until the error is no longer infinity.
    auto start(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> void;
};

} // namespace Optima
