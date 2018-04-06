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
#include <Optima/Matrix.hpp>

namespace Optima {

/// Define a type that represents an index.
using Index = std::ptrdiff_t;

/// Define a type that represents a vector of indices.
using Indices = Eigen::Matrix<Index, Eigen::Dynamic, 1>;

/// Define a type that represents a reference to a vector of indices.
using IndicesRef = Eigen::Ref<Indices>;

/// Define a type that represents a constant reference to a vector of indices.
using IndicesConstRef = Eigen::Ref<const Indices>;

} // namespace Optima
