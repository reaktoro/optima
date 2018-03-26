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
#include <Optima/OptimumState.hpp>
#include <Optima/VariantMatrix.hpp>

namespace Optima {

/// The requirements in the evaluation of the objective function.
class ObjectiveRequirement
{
public:
    /// The boolean flag that indicates the need for the objective value.
    bool f = true;

    /// The boolean flag that indicates the need for the objective gradient.
    bool g = true;

    /// The boolean flag that indicates the need for the objective Hessian.
    bool H = true;
};

/// The evaluated state of an objective function.
class ObjectiveState
{
public:
    /// The evaluated value of the objective function.
    double& f;

    /// The evaluated gradient of the objective function.
    VectorRef g;

    /// The evaluated Hessian of the objective function.
    /// The dimension of the Hessian matrix is controlled using the methods:
    /// OptimumStructure::setHessianMatrixAsDense,
    /// OptimumStructure::setHessianMatrixAsDiagonal, and
    /// OptimumStructure::setHessianMatrixAsZero.
    /// The first method causes `hessian` to be a square matrix with dimensions
    /// of the number of variables in the optimization problem. The second method
    /// causes `hessian` to be a matrix with a single column and as many rows
    /// as there are variables. The third method causes `hessian` to be an empty
    /// matrix, with zero rows and columns.
    VariantMatrixRef H;

    /// The requirements in the evaluation of the objective function.
    ObjectiveRequirement requires;

    /// The boolean flag that indicates if the objective function evaluation failed.
    bool failed;

    /// Construct an ObjectiveState instance.
    /// @param grad The vector reference to store the gradient calculation.
    /// @param hessian The matrix reference to store the Hessian calculation.
    ObjectiveState(OptimumState& state)
    : f(state.f), g(state.g), H(state.H), failed(false)
    {}
};

} // namespace Optima
