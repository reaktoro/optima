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

// Optima includes
#include <Optima/Matrix.hpp>

namespace Optima {

/// Define a type that represents an index.
using Index = Eigen::Index;

/// Define a type that represents a vector of indices.
using Indices = Eigen::VectorXl;

/// Define a type that represents a reference to a vector of indices.
using IndicesRef = Eigen::VectorXlRef;

/// Define a type that represents a constant reference to a vector of indices.
using IndicesView = Eigen::VectorXlView;

} // namespace Optima
