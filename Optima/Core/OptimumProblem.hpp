//// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
////
//// Copyright (C) 2014-2017 Allan Leal
////
//// This program is free software: you can redistribute it and/or modify
//// it under the terms of the GNU General Public License as published by
//// the Free Software Foundation, either version 3 of the License, or
//// (at your option) any later version.
////
//// This program is distributed in the hope that it will be useful,
//// but WITHOUT ANY WARRANTY; without even the implied warranty of
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//// GNU General Public License for more details.
////
//// You should have received a copy of the GNU General Public License
//// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
//#pragma once
//
//// Optima includes
//#include <Optima/Core/OptimumParams.hpp>
//#include <Optima/Core/OptimumStructure.hpp>
//
//namespace Optima {
//
///// The parameters of an optimization problem that change with more frequency.
//class OptimumProblem
//{
//public:
//    /// Construct a default OptimumProblem instance.
//    /// @param structure The structure of the optimization problem.
//    OptimumProblem(const OptimumStructure& structure);
//
//    /// Return right-hand side vector of the equality constraint \eq{Ax = b}.
//    auto b() -> VectorXdRef { return m_b; }
//
//    /// Return right-hand side vector of the equality constraint \eq{Ax = b}.
//    auto b() const -> VectorXdConstRef { return m_b; }
//
//    /// Set a common lower bound value for the variables \eq{x}.
//    auto xlower(double val) -> void;
//
//    /// Set the lower bounds of the variables \eq{x}.
//    /// @param values The values of the lower bounds.
//    auto xlower(VectorXdConstRef values) -> void;
//
//    /// Set the lower bounds of selected variables in \eq{x}.
//    /// @param indices The indices of the variables in \eq{x} with lower bounds.
//    /// @param values The values of the lower bounds.
//    auto xlower(VectorXiConstRef indices, VectorXdConstRef values) -> void;
//
//    /// Return the lower bounds of the variables \eq{x}.
//    auto xlower() const -> VectorXdConstRef;
//
//    /// Set a common upper bound value for the variables \eq{x}.
//    auto xupper(double val) -> void;
//
//    /// Set the upper bounds of the variables \eq{x}.
//    /// @param values The values of the upper bounds.
//    auto xupper(VectorXdConstRef values) -> void;
//
//    /// Set the upper bounds of selected variables in \eq{x}.
//    /// @param indices The indices of the variables in \eq{x} with upper bounds.
//    /// @param values The values of the upper bounds.
//    auto xupper(VectorXiConstRef indices, VectorXdConstRef values) -> void;
//
//    /// Return the upper bounds of the variables \eq{x}.
//    auto xupper() const -> VectorXdConstRef;
//
//    /// Set a common value for the fixed variables in \eq{x}.
//    auto xfixed(double val) -> void;
//
//    /// Set the values of the fixed variables in \eq{x}.
//    /// @param values The values of the fixed variables.
//    auto xfixed(VectorXdConstRef values) -> void;
//
//    /// Set the fixed values of selected variables in \eq{x}.
//    /// @param indices The indices of the fixed variables in \eq{x}.
//    /// @param values The values of the fixed variables.
//    auto xfixed(VectorXiConstRef indices, VectorXdConstRef values) -> void;
//
//    /// Return the values of the fixed variables in \eq{x}.
//    auto xfixed() const -> VectorXdConstRef;
//
//    /// Return the indices of the variables with lower bounds.
//    auto iwithlower() const -> VectorXiConstRef;
//
//    /// Return the indices of the variables with upper bounds.
//    auto iwithupper() const -> VectorXiConstRef;
//
//    /// Return the indices of the variables with fixed values.
//    auto iwithfixed() const -> VectorXiConstRef;
//
//    /// Return the indices of the variables without lower bounds.
//    auto iwithoutlower() const -> VectorXiConstRef;
//
//    /// Return the indices of the variables without upper bounds.
//    auto iwithoutupper() const -> VectorXiConstRef;
//
//    /// Return the indices of the variables without fixed values.
//    auto iwithoutfixed() const -> VectorXiConstRef;
//
//    /// Return the indices of the variables partitioned in [with, without] lower bounds.
//    auto lowerpartition() const -> VectorXiConstRef;
//
//    /// Return the indices of the variables partitioned in [with, without] upper bounds.
//    auto upperpartition() const -> VectorXiConstRef;
//
//    /// Return the indices of the variables partitioned in [with, without] fixed values.
//    auto fixedpartition() const -> VectorXiConstRef;
//
//private:
//    /// The number of variables.
//    Index m_n;
//
//    /// The number of equality constraints.
//    Index m_m;
//
//    /// The number of variables with lower bounds.
//    Index m_nlower;
//
//    /// The number of variables with upper bounds.
//    Index m_nupper;
//
//    /// The number of variables with fixed values.
//    Index m_nfixed;
//
//    /// The right-hand side vector of the linear equality constraint \eq{Ax = b}.
//    VectorXd m_b;
//
//    /// The indices of the variables partitioned in [with, without] lower bounds.
//    VectorXi m_lowerpartition;
//
//    /// The indices of the variables partitioned in [with, without] upper bounds.
//    VectorXi m_upperpartition;
//
//    /// The indices of the variables partitioned in [with, without] fixed values.
//    VectorXi m_fixedpartition;
//
//    /// The lower bounds of the variables \eq{x}.
//    VectorXd m_xlower;
//
//    /// The upper bounds of the variables \eq{x}.
//    VectorXd m_xupper;
//
//    /// The values of the variables in \eq{x} that are fixed.
//    VectorXd m_xfixed;
//
//};
//
//} // namespace Optima
