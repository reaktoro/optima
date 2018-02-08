//// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
////
//// Copyright (C) 2014-2018 Allan Leal
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
//#include "OptimumProblem.hpp"
//
//namespace Optima {
//
//
//OptimumProblem::OptimumProblem(const OptimumStructure& structure)
//: m_n(structure.n), m_m(structure.A.rows()),
//  m_nlower(0), m_nupper(0), m_nfixed(0), m_b(m_m),
//  m_xlower(m_n), m_xupper(m_n), m_xfixed(m_n)
//{
//    m_lowerpartition.setLinSpaced(m_n, 0, m_n - 1);
//    m_upperpartition.setLinSpaced(m_n, 0, m_n - 1);
//    m_fixedpartition.setLinSpaced(m_n, 0, m_n - 1);
//    m_xlower.fill(0.0);
//    m_xupper.fill(0.0);
//    m_xfixed.fill(0.0);
//}
//
//auto OptimumProblem::xlower(double val) -> void
//{
//    m_xlower.fill(val);
//    m_nlower = m_n;
//}
//
//auto OptimumProblem::xlower(VectorXdConstRef values) -> void
//{
//    m_xlower.head(m_n) = values;
//    m_nlower = m_n;
//}
//
//auto OptimumProblem::xlower(VectorXiConstRef indices, VectorXdConstRef values) -> void
//{
//    m_xlower(indices) = values;
//    m_nlower = indices.size();
//    m_lowerpartition.setLinSpaced(m_n, 0, m_n - 1);
//    m_lowerpartition.head(m_nlower).swap(m_lowerpartition(indices));
//}
//
//auto OptimumProblem::xlower() const -> VectorXdConstRef
//{
//    return m_xlower.head(m_nlower);
//}
//
//auto OptimumProblem::xupper(double val) -> void
//{
//    m_xupper.fill(val);
//    m_nupper = m_n;
//}
//
//auto OptimumProblem::xupper(VectorXdConstRef values) -> void
//{
//    m_xupper.head(m_n) = values;
//    m_nupper = m_n;
//}
//
//auto OptimumProblem::xupper(VectorXiConstRef indices, VectorXdConstRef values) -> void
//{
//    m_xupper(indices) = values;
//    m_nupper = indices.size();
//    m_upperpartition.setLinSpaced(m_n, 0, m_n - 1);
//    m_upperpartition.head(m_nlower).swap(m_upperpartition(indices));
//}
//
//auto OptimumProblem::xupper() const -> VectorXdConstRef
//{
//    return m_xupper.head(m_nupper);
//}
//
//auto OptimumProblem::xfixed(double val) -> void
//{
//    m_xfixed.fill(val);
//    m_nfixed = m_n;
//}
//
//auto OptimumProblem::xfixed(VectorXdConstRef values) -> void
//{
//    m_xfixed.head(m_n) = values;
//    m_nfixed = m_n;
//}
//
//auto OptimumProblem::xfixed(VectorXiConstRef indices, VectorXdConstRef values) -> void
//{
//    m_xfixed(indices) = values;
//    m_nfixed = indices.size();
//    m_fixedpartition.setLinSpaced(m_n, 0, m_n - 1);
//    m_fixedpartition.head(m_nlower).swap(m_fixedpartition(indices));
//}
//
//auto OptimumProblem::xfixed() const -> VectorXdConstRef
//{
//    return m_xfixed.head(m_nfixed);
//}
//
//auto OptimumProblem::iwithlower() const -> VectorXiConstRef
//{
//    return m_lowerpartition.head(m_nlower);
//}
//
//auto OptimumProblem::variablesWithUpperBounds() const -> VectorXiConstRef
//{
//    return m_upperpartition.head(m_nupper);
//}
//
//auto OptimumProblem::variablesWithFixedValues() const -> VectorXiConstRef
//{
//    return m_fixedpartition.head(m_nfixed);
//}
//
//auto OptimumProblem::variablesWithoutLowerBounds() const -> VectorXiConstRef
//{
//    return m_lowerpartition.tail(m_n - m_nlower);
//}
//
//auto OptimumProblem::variablesWithoutUpperBounds() const -> VectorXiConstRef
//{
//    return m_upperpartition.tail(m_n - m_nupper);
//}
//
//auto OptimumProblem::variablesWithoutFixedValues() const -> VectorXiConstRef
//{
//    return m_fixedpartition.tail(m_n - m_nfixed);
//}
//
//auto OptimumProblem::lowerpartition() const -> VectorXiConstRef
//{
//    return m_lowerpartition;
//}
//
//auto OptimumProblem::upperpartition() const -> VectorXiConstRef
//{
//    return m_upperpartition;
//}
//
//auto OptimumProblem::fixedpartition() const -> VectorXiConstRef
//{
//    return m_fixedpartition;
//}
//
//} // namespace Optima
