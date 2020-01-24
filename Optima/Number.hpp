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
#include <Optima/Index.hpp>

namespace Optima {

/// A wrapper for a number to be used as reference in functions.
template<typename T>
struct Number
{
    T value = {};
    Number() {}
    Number(const T& val) : value(val) {}
    auto operator=(const T& val) { value = val; return *this; }
    operator T() const { return value; }
};

using DoubleNumber = Number<double>;
using DoubleNumberRef = DoubleNumber&;

using IndexNumber = Number<Index>;
using IndexNumberRef = IndexNumber&;

} // namespace Optima
