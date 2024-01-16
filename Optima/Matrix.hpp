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

// Eigen includes
#include <Optima/Eigen.hpp>

namespace Optima {

#include <Optima/deps/eigenx/Eigen/Typedefs>

/// An Eigen matrix type to be used with numpy, which expects row-major order.
/// See this discussion [here](https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders).
using Matrix4py = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, Eigen::Dynamic, Eigen::Dynamic>;

/// An Eigen reference matrix type to be used with numpy arrays.
/// See this discussion [here](https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders).
using MatrixRef4py = Eigen::Ref<Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

/// An Eigen constant reference matrix type to be used with numpy arrays.
/// See this discussion [here](https://pybind11.readthedocs.io/en/stable/advanced/cast/eigen.html#storage-orders).
using MatrixView4py = Eigen::Ref<const Eigen::MatrixXd, 0, Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>>;

} // namespace Optima
