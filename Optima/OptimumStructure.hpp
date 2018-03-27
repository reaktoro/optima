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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

/// The structure of an optimization problem that changes with less frequency.
class OptimumStructure
{
public:
    /// Construct an OptimumStructure instance without equality constraints.
    /// @param n The number of variables in \eq{x} in the optimization problem.
    OptimumStructure(Index n);

    /// Construct an OptimumStructure instance with equality constraints.
    /// @param n The number of variables in \eq{x} in the optimization problem.
    /// @param m The number of linear equality constraints in the optimization problem.
    OptimumStructure(Index n, Index m);

    /// Construct an OptimumStructure instance with equality constraints.
    /// @param A The linear equality constraint matrix \eq{A} in the optimization problem.
    OptimumStructure(MatrixConstRef A);

    /// Set the indices of the variables in \eq{x} with lower bounds.
    auto setVariablesWithLowerBounds(VectorXiConstRef indices) -> void;

    /// Set all variables in \eq{x} with lower bounds.
    auto allVariablesHaveLowerBounds() -> void;

    /// Set the indices of the variables in \eq{x} with upper bounds.
    auto setVariablesWithUpperBounds(VectorXiConstRef indices) -> void;

    /// Set all variables in \eq{x} with upper bounds.
    auto allVariablesHaveUpperBounds() -> void;

    /// Set the indices of the variables in \eq{x} with fixed values.
    auto setVariablesWithFixedValues(VectorXiConstRef indices) -> void;

    /// Set the structure of the Hessian matrix to be dense.
    auto setHessianMatrixAsDense() -> void { _structure_hessian_matrix = MatrixStructure::Dense; }

    /// Set the structure of the Hessian matrix to be diagonal.
    auto setHessianMatrixAsDiagonal() -> void { _structure_hessian_matrix = MatrixStructure::Diagonal; }

    /// Set the structure of the Hessian matrix to be fully zero.
    auto setHessianMatrixAsZero() -> void { _structure_hessian_matrix = MatrixStructure::Zero; }

    /// Return the number of variables.
    auto numVariables() const -> Index { return _n; }

    /// Return the number of linear equality constraints.
    auto numEqualityConstraints() const -> Index { return _A.rows(); }

    /// Return the indices of the variables with lower bounds.
    auto variablesWithLowerBounds() const -> VectorXiConstRef { return _lowerpartition.tail(_nlower); }

    /// Return the indices of the variables with upper bounds.
    auto variablesWithUpperBounds() const -> VectorXiConstRef { return _upperpartition.tail(_nupper); }

    /// Return the indices of the variables with fixed values.
    auto variablesWithFixedValues() const -> VectorXiConstRef { return _fixedpartition.tail(_nfixed); }

    /// Return the indices of the variables without lower bounds.
    auto variablesWithoutLowerBounds() const -> VectorXiConstRef { return _lowerpartition.head(_n - _nlower); }

    /// Return the indices of the variables without upper bounds.
    auto variablesWithoutUpperBounds() const -> VectorXiConstRef { return _upperpartition.head(_n - _nupper); }

    /// Return the indices of the variables without fixed values.
    auto variablesWithoutFixedValues() const -> VectorXiConstRef { return _fixedpartition.head(_n - _nfixed); }

    /// Return the indices of the variables partitioned in [without, with] lower bounds.
    auto orderingLowerBounds() const -> VectorXiConstRef { return _lowerpartition; }

    /// Return the indices of the variables partitioned in [without, with] upper bounds.
    auto orderingUpperBounds() const -> VectorXiConstRef { return _upperpartition; }

    /// Return the indices of the variables partitioned in [without, with] fixed values.
    auto orderingFixedValues() const -> VectorXiConstRef { return _fixedpartition; }

    /// Return the structure type of the Hessian matrix.
    auto structureHessianMatrix() const -> MatrixStructure { return _structure_hessian_matrix; }

    /// Return the coefficient matrix \eq{A} of the linear equality constraints.
    auto equalityConstraintMatrix() const -> MatrixConstRef { return _A; }

    /// Return the coefficient matrix \eq{A} of the linear equality constraints.
    auto equalityConstraintMatrix() -> MatrixRef { return _A; }

private:
    /// The number of variables in the optimization problem.
    Index _n;

    /// The number of linear equality constraints in the optimization problem.
    Index _m;

    /// The coefficient matrix of the linear equality constraint \eq{Ax = a}.
    Matrix _A;

    /// The number of variables with lower bounds.
    Index _nlower;

    /// The number of variables with upper bounds.
    Index _nupper;

    /// The number of variables with fixed values.
    Index _nfixed;

    /// The indices of the variables partitioned in [with, without] lower bounds.
    VectorXi _lowerpartition;

    /// The indices of the variables partitioned in [with, without] upper bounds.
    VectorXi _upperpartition;

    /// The indices of the variables partitioned in [with, without] fixed values.
    VectorXi _fixedpartition;

    /// The structure of the Hessian matrix
    MatrixStructure _structure_hessian_matrix;
};

} // namespace Optima
