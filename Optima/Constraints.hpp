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
#include <Optima/Matrix.hpp>

namespace Optima {

/// The result of the evaluation of a constraint function.
struct ConstraintResult
{
    /// The value of the evaluated constraint function.
    Vector value;

    /// The Jacobian matrix of the evaluated constraint function.
    Matrix jacobian;
};

/// The signature of a constraint function.
using ConstraintFunction = std::function<void(VectorConstRef, ConstraintResult&)>;

/// The constraints in an optimization problem.
class Constraints
{
public:
    /// Construct a default Constraints instance.
    Constraints();

    /// Construct a Constraints instance with given number of variables.
    /// @param n The number of variables in \eq{x} in the optimization problem.
    explicit Constraints(Index n);

    /// Set the matrix \eq{A_e} of the linear equality constraints.
    auto setEqualityConstraintMatrix(MatrixConstRef Ae) -> void;

    /// Set the non-linear equality constraint function \eq{h_e(x)}.
    /// @param he The constraint function in the non-linear equality constraint equations.
    /// @param m_he The number of non-linear equality constraint equations.
    auto setEqualityConstraintFunction(const ConstraintFunction& he, Index m_he) -> void;

    /// Set the matrix \eq{A_i} of the linear inequality constraints.
    auto setInequalityConstraintMatrix(MatrixConstRef Ai) -> void;

    /// Set the non-linear equality constraint function \eq{h_i(x)}.
    /// @param hi The constraint function in the non-linear inequality constraint equations.
    /// @param m_nhi The number of non-linear inequality constraint equations.
    auto setInequalityConstraintFunction(const ConstraintFunction& hi, Index m_nhi) -> void;

    /// Set the indices of the variables in \eq{x} with lower bounds.
    auto setVariablesWithLowerBounds(IndicesConstRef indices) -> void;

    /// Set all variables in \eq{x} with lower bounds.
    auto allVariablesHaveLowerBounds() -> void;

    /// Set the indices of the variables in \eq{x} with upper bounds.
    auto setVariablesWithUpperBounds(IndicesConstRef indices) -> void;

    /// Set all variables in \eq{x} with upper bounds.
    auto allVariablesHaveUpperBounds() -> void;

    /// Set the indices of the variables in \eq{x} with fixed values.
    auto setVariablesWithFixedValues(IndicesConstRef indices) -> void;

    /// Return the number of variables.
    auto numVariables() const -> Index;

    /// Return the number of linear equality constraints.
    auto numLinearEqualityConstraints() const -> Index;

    /// Return the number of linear inequality constraints.
    auto numLinearInequalityConstraints() const -> Index;

    /// Return the number of non-linear equality constraints.
    auto numNonLinearEqualityConstraints() const -> Index;

    /// Return the number of non-linear inequality constraints.
    auto numNonLinearInequalityConstraints() const -> Index;

    /// Return the equality constraint matrix \eq{A_e}.
    auto equalityConstraintMatrix() const -> MatrixConstRef;

    /// Return the equality constraint function \eq{h_{e}(x)}.
    auto equalityConstraintFunction() const -> const ConstraintFunction&;

    /// Return the inequality constraint matrix \eq{A_i}.
    auto inequalityConstraintMatrix() const -> MatrixConstRef;

    /// Return the inequality constraint function \eq{h_{i}(x)}.
    auto inequalityConstraintFunction() const -> const ConstraintFunction&;

    /// Return the indices of the variables with lower bounds.
    auto variablesWithLowerBounds() const -> IndicesConstRef;

    /// Return the indices of the variables with upper bounds.
    auto variablesWithUpperBounds() const -> IndicesConstRef;

    /// Return the indices of the variables with fixed values.
    auto variablesWithFixedValues() const -> IndicesConstRef;

    /// Return the indices of the variables without lower bounds.
    auto variablesWithoutLowerBounds() const -> IndicesConstRef;

    /// Return the indices of the variables without upper bounds.
    auto variablesWithoutUpperBounds() const -> IndicesConstRef;

    /// Return the indices of the variables without fixed values.
    auto variablesWithoutFixedValues() const -> IndicesConstRef;

    /// Return the indices of the variables partitioned in [without, with] lower bounds.
    auto orderingLowerBounds() const -> IndicesConstRef;

    /// Return the indices of the variables partitioned in [without, with] upper bounds.
    auto orderingUpperBounds() const -> IndicesConstRef;

    /// Return the indices of the variables partitioned in [without, with] fixed values.
    auto orderingFixedValues() const -> IndicesConstRef;

private:
    /// The number of variables in the optimization problem.
    Index n = 0;

    /// The coefficient matrix of the linear equality constraint equations \eq{A_{e}x=b_{e}}.
    Matrix Ae;

    /// The coefficient matrix of the linear inequality constraint equations \eq{A_{i}x\geq b_{i}}.
    Matrix Ai;

    /// The constraint function in the non-linear equality constraint equations \eq{h_{e}(x) = 0}.
    ConstraintFunction he;

    /// The constraint function in the non-linear inequality constraint equations \eq{h_{i}(x) \geq 0}.
    ConstraintFunction hi;

    /// The number of non-linear equality constraint equations.
    Index m_he = 0;

    /// The number of non-linear inequality constraint equations.
    Index m_hi = 0;

    /// The number of variables with lower bounds.
    Index nlower = 0;

    /// The number of variables with upper bounds.
    Index nupper = 0;

    /// The number of variables with fixed values.
    Index nfixed = 0;

    /// The indices of the variables partitioned in [with, without] lower bounds.
    Indices lowerpartition;

    /// The indices of the variables partitioned in [with, without] upper bounds.
    Indices upperpartition;

    /// The indices of the variables partitioned in [with, without] fixed values.
    Indices fixedpartition;
};

} // namespace Optima
