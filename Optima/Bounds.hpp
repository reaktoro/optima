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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent the lower and upper bounds of variables.
class Bounds
{
public:
    /// Construct a default Bounds instance.
    Bounds(Index size);

    /// Construct a Bounds instance with given lower and upper bounds.
    Bounds(VectorConstRef lower, VectorConstRef upper);

    /// Construct a copy of a Bounds instance.
    Bounds(const Bounds& other);

    /// Destroy this Bounds instance.
    virtual ~Bounds();

    /// Assign a Bounds instance to this.
    auto operator=(Bounds other) -> Bounds&;

    /// Return the lower bounds of the variables.
    auto lower() const -> VectorConstRef;

    /// Return the upper bounds of the variables.
    auto upper() const -> VectorConstRef;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
