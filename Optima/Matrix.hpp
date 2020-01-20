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

// Eigen includes
#include <Optima/deps/eigen3/Eigen/Core>
#include <Optima/deps/eigenx/Eigen/Functions>
#include <Optima/deps/eigenx/Eigen/Types>

namespace Optima {

#include <Optima/deps/eigenx/Eigen/Typedefs>

/// An alias for a more general Eigen matrix reference to be used in Python.
/// The need for this special type is due to the different order in which Eigen matrices
/// and numpy arrays arrange their memory, with Eigen using column-major order and numpy
/// using row-major order. See this discussion [here](https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders).
using MatrixRef4py = Eigen::Ref<Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

using MatrixConstRef4py = Eigen::Ref<const Matrix, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

} // namespace Optima
