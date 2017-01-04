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

// Eigenx includes
#include <Eigenx/Core.hpp>

namespace Optima {

/// Alias to the vector type of the Eigen library.
using VectorXd = Eigen::VectorXd;

/// Alias to the matrix type of the Eigen library.
using MatrixXd = Eigen::MatrixXd;

/// Alias to a reference to a vector type of the Eigen library.
using VectorRef = Eigen::Ref<VectorXd>;

/// Alias to a reference to a matrix type of the Eigen library.
using MatrixRef = Eigen::Ref<MatrixXd>;

/// Alias to a const reference to a vector type of the Eigen library.
using ConstVectorRef = Eigen::Ref<const VectorXd>;

/// Alias to a const reference to a matrix type of the Eigen library.
using ConstMatrixRef = Eigen::Ref<const MatrixXd>;

/// Define an alias to a permutation matrix type of the Eigen library
using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

} // namespace Optima

