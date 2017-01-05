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

namespace Optima {

/// Used to create optional variables that are needed in the form of constant references.
template<typename Value>
class Optional
{
public:
    /// Construct a default Optional instance.
    Optional() : initialized(false), dummy(), ref(dummy) {}

    /// Construct a Optional instance with given value.
    Optional(const Value& value) : initialized(true), dummy(), ref(value) {}

    /// Construct a Optional instance with given value.
    Optional(const Optional& other) : initialized(other.initialized), dummy(), ref(other.initialized ? other.ref : dummy) {}

    /// Return the value that the Optional instance holds.
    auto value() const -> const Value& { return initialized ? ref : dummy; }

    /// Return `true` if the Optional instance is empty.
    auto empty() const -> bool { return !initialized; }

    /// Convert this Optional instance into a `bool`
    operator bool() const { return initialized; }

private:
    /// The boolean flag that indicates if the Optional instance was initialized with a value.
    bool initialized = false;

    /// The dummy default value to be returned in @ref value if none was given.
    Value dummy;

    /// The reference to a given value or to @ref dummy.
    const Value& ref;
};

} // namespace Optima
