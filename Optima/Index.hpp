// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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
#include <Optima/Matrix.hpp>

namespace Optima {

/// Define a type that represents an index
using Index = std::ptrdiff_t;

/// Return a vector of indices with values from 0 up to a given length.
inline auto indices(Index length) -> VectorXi
{
    VectorXi res(length);
    for(Index i = 0; i < length; ++i) res[i] = i;
    return res;
}

} // namespace Optima
