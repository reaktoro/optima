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

#include "OptimumStructure.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Matrix.hpp>
using namespace Eigen;

namespace Optima {

ObjectiveState::ObjectiveState(VectorRef grad, MatrixRef hessian)
: grad(grad), hessian(hessian), failed(false)
{}

OptimumStructure::OptimumStructure(ObjectiveFunction f, Index n, Index m)
: m_objective(f), m_n(n), m_m(m), m_A(zeros(m_m, m_n)),
  m_nlower(0), m_nupper(0), m_nfixed(0),
  m_lowerpartition(linspace<int>(n, 0, n - 1)),
  m_upperpartition(linspace<int>(n, 0, n - 1)),
  m_fixedpartition(linspace<int>(n, 0, n - 1)),
  m_structure_hessian_matrix(Dense)
{}

OptimumStructure::OptimumStructure(ObjectiveFunction f, MatrixConstRef A)
: OptimumStructure(f, A.cols(), A.rows())
{
    m_A = A;
}

auto OptimumStructure::setEqualityConstraintMatrix(MatrixConstRef A) -> void
{
    MatrixRef Aref(m_A);
    Aref = A;
}

auto OptimumStructure::setVariablesWithLowerBounds(VectorXiConstRef indices) -> void
{
    m_nlower = indices.size();
    m_lowerpartition.setLinSpaced(m_n, 0, m_n - 1);
    m_lowerpartition.tail(m_nlower).swap(m_lowerpartition(indices));
}

auto OptimumStructure::allVariablesHaveLowerBounds() -> void
{
    m_nlower = m_n;
    m_lowerpartition.setLinSpaced(m_n, 0, m_n - 1);
}

auto OptimumStructure::setVariablesWithUpperBounds(VectorXiConstRef indices) -> void
{
    m_nupper = indices.size();
    m_upperpartition.setLinSpaced(m_n, 0, m_n - 1);
    m_upperpartition.tail(m_nupper).swap(m_upperpartition(indices));
}

auto OptimumStructure::allVariablesHaveUpperBounds() -> void
{
    m_nupper = m_n;
    m_upperpartition.setLinSpaced(m_n, 0, m_n - 1);
}

auto OptimumStructure::setVariablesWithFixedValues(VectorXiConstRef indices) -> void
{
    m_nfixed = indices.size();
    m_fixedpartition.setLinSpaced(m_n, 0, m_n - 1);
    m_fixedpartition.tail(m_nfixed).swap(m_fixedpartition(indices));
}

} // namespace Optima
