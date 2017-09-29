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

// Optima includes
#include <Optima/Common/Index.hpp>
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// The requirements in the evaluation of the objective function.
class ObjectiveRequirement
{
public:
    /// The boolean flag that indicates the need for the objective value.
    bool val = true;

    /// The boolean flag that indicates the need for the objective gradient.
    bool grad = true;

    /// The boolean flag that indicates the need for the objective Hessian.
    bool hessian = true;
};

/// The evaluated state of an objective function.
class ObjectiveState
{
public:
    /// The evaluated value of the objective function.
    double val;

    /// The evaluated gradient of the objective function.
    VectorXd grad;

    /// The evaluated Hessian of the objective function.
    MatrixXd hessian;

    /// The requirements in the evaluation of the objective function.
    ObjectiveRequirement requires;

    /// The boolean flag that indicates if the objective function evaluation failed.
    bool failed = false;
};

/// The functional signature of an objective function.
/// @param x The values of the variables \eq{x}.
/// @param f The evaluated state of the objective function.
using ObjectiveFunction = std::function<void(VectorXdConstRef, ObjectiveState&)>;

// todo Implement this as a class using the pimpl idiom
// OptimumStructure structure;
// structure.setNumVariables(10);
// structure.setNumEqualityConstraints(5);
// structure.setNumInequalityConstraints(3);
// structure.setConstraintMatrix(A);
// structure.setObjective(obj);
/// The structure of an optimization problem that changes with less frequency.
class OptimumStructure
{
public:
    /// Construct an OptimumStructure instance.
    OptimumStructure(Index n);

    /// Set the indices of the variables in \eq{x} with lower bounds.
    auto withLowerBounds(VectorXiConstRef indices) -> void;

    /// Set all variables in \eq{x} with lower bounds.
    auto withLowerBounds() -> void;

    /// Set the indices of the variables in \eq{x} with upper bounds.
    auto withUpperBounds(VectorXiConstRef indices) -> void;

    /// Set all variables in \eq{x} with upper bounds.
    auto withUpperBounds() -> void;

    /// Set the indices of the variables in \eq{x} with fixed values.
    auto withFixedValues(VectorXiConstRef indices) -> void;

    /// Return the indices of the variables with lower bounds.
    auto iwithlower() const -> VectorXiConstRef { return m_lowerpartition.tail(m_nlower); }

    /// Return the indices of the variables with upper bounds.
    auto iwithupper() const -> VectorXiConstRef { return m_upperpartition.tail(m_nupper); }

    /// Return the indices of the variables with fixed values.
    auto iwithfixed() const -> VectorXiConstRef { return m_fixedpartition.tail(m_nfixed); }

    /// Return the indices of the variables without lower bounds.
    auto iwithoutlower() const -> VectorXiConstRef { return m_lowerpartition.head(n - m_nlower); }

    /// Return the indices of the variables without upper bounds.
    auto iwithoutupper() const -> VectorXiConstRef { return m_upperpartition.head(n - m_nupper); }

    /// Return the indices of the variables without fixed values.
    auto iwithoutfixed() const -> VectorXiConstRef { return m_fixedpartition.head(n - m_nfixed); }

    /// Return the indices of the variables partitioned in [without, with] lower bounds.
    auto lowerpartition() const -> VectorXiConstRef { return m_lowerpartition; }

    /// Return the indices of the variables partitioned in [without, with] upper bounds.
    auto upperpartition() const -> VectorXiConstRef { return m_upperpartition; }

    /// Return the indices of the variables partitioned in [without, with] fixed values.
    auto fixedpartition() const -> VectorXiConstRef { return m_fixedpartition; }

//private:
    /// The number of variables in the optimization problem.
    Index n;

    /// The objective function of the optimization problem.
    ObjectiveFunction objective;

    /// The coefficient matrix of the linear equality constraint \eq{Ax = a}.
    MatrixXd A;

    /// The number of variables with lower bounds.
    Index m_nlower;

    /// The number of variables with upper bounds.
    Index m_nupper;

    /// The number of variables with fixed values.
    Index m_nfixed;

    /// The indices of the variables partitioned in [with, without] lower bounds.
    VectorXi m_lowerpartition;

    /// The indices of the variables partitioned in [with, without] upper bounds.
    VectorXi m_upperpartition;

    /// The indices of the variables partitioned in [with, without] fixed values.
    VectorXi m_fixedpartition;
};

} // namespace Optima
