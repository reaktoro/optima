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
#include <eigenx/Eigen/Core>

namespace Optima {

using Vector = Eigen::VectorXd; ///< Alias to Eigen type Eigen::VectorXd.
using VectorXd = Eigen::VectorXd; ///< Alias to Eigen type Eigen::VectorXd.
using VectorXi = Eigen::VectorXi; ///< Alias to Eigen type Eigen::VectorXi.

using VectorRef = Eigen::Ref<VectorXd>; ///< Alias to Eigen type Eigen::Ref<VectorXd>.
using VectorXdRef = Eigen::Ref<VectorXd>; ///< Alias to Eigen type Eigen::Ref<VectorXd>.
using VectorXiRef = Eigen::Ref<VectorXi>; ///< Alias to Eigen type Eigen::Ref<VectorXi>.

using VectorConstRef = Eigen::Ref<const VectorXd>; ///< Alias to Eigen type Eigen::Ref<const VectorXd>.
using VectorXdConstRef = Eigen::Ref<const VectorXd>; ///< Alias to Eigen type Eigen::Ref<const VectorXd>.
using VectorXiConstRef = Eigen::Ref<const VectorXi>; ///< Alias to Eigen type Eigen::Ref<const VectorXi>.

using VectorMap = Eigen::Map<VectorXd>; ///< Alias to Eigen type Eigen::Map<VectorXd>.
using VectorXdMap = Eigen::Map<VectorXd>; ///< Alias to Eigen type Eigen::Map<VectorXd>.
using VectorXiMap = Eigen::Map<VectorXi>; ///< Alias to Eigen type Eigen::Map<VectorXi>.

using VectorConstMap = Eigen::Map<const VectorXd>; ///< Alias to Eigen type Eigen::Map<const VectorXd>.
using VectorXdConstMap = Eigen::Map<const VectorXd>; ///< Alias to Eigen type Eigen::Map<const VectorXd>.
using VectorXiConstMap = Eigen::Map<const VectorXi>; ///< Alias to Eigen type Eigen::Map<const VectorXi>.

using Matrix = Eigen::MatrixXd; ///< Alias to Eigen type Eigen::MatrixXd.
using MatrixXd = Eigen::MatrixXd; ///< Alias to Eigen type Eigen::MatrixXd.
using MatrixXi = Eigen::MatrixXi; ///< Alias to Eigen type Eigen::MatrixXi.

using MatrixRef = Eigen::Ref<MatrixXd>; ///< Alias to Eigen type Eigen::Ref<MatrixXd>.
using MatrixXdRef = Eigen::Ref<MatrixXd>; ///< Alias to Eigen type Eigen::Ref<MatrixXd>.
using MatrixXiRef = Eigen::Ref<MatrixXi>; ///< Alias to Eigen type Eigen::Ref<MatrixXi>.

using MatrixConstRef = Eigen::Ref<const MatrixXd>; ///< Alias to Eigen type Eigen::Ref<const MatrixXd>.
using MatrixXdConstRef = Eigen::Ref<const MatrixXd>; ///< Alias to Eigen type Eigen::Ref<const MatrixXd>.
using MatrixXiConstRef = Eigen::Ref<const MatrixXi>; ///< Alias to Eigen type Eigen::Ref<const MatrixXi>.

using MatrixMap = Eigen::Map<MatrixXd>; ///< Alias to Eigen type Eigen::Map<MatrixXd>.
using MatrixXdMap = Eigen::Map<MatrixXd>; ///< Alias to Eigen type Eigen::Map<MatrixXd>.
using MatrixXiMap = Eigen::Map<MatrixXi>; ///< Alias to Eigen type Eigen::Map<MatrixXi>.

using MatrixConstMap = Eigen::Map<const MatrixXd>; ///< Alias to Eigen type Eigen::Map<const MatrixXd>.
using MatrixXdConstMap = Eigen::Map<const MatrixXd>; ///< Alias to Eigen type Eigen::Map<const MatrixXd>.
using MatrixXiConstMap = Eigen::Map<const MatrixXi>; ///< Alias to Eigen type Eigen::Map<const MatrixXi>.

using PermutationMatrix = Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>; ///< /// Alias to a permutation matrix type of the Eigen library.

} // namespace Optima

