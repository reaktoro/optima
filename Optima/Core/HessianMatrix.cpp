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

#include "HessianMatrix.hpp"

namespace Optima {

HessianBlock::HessianBlock()
: m_dim(0), m_mode(Zero)
{}

HessianBlock::~HessianBlock()
{}

auto HessianBlock::zero(Index dim) -> void
{
    m_dim = dim;
    m_mode = Zero;
}

auto HessianBlock::diagonal() -> Vector&
{
    m_mode = Diagonal;
    return m_diagonal;
}

auto HessianBlock::diagonal() const -> const Vector&
{
    return m_diagonal;
}

auto HessianBlock::dense() -> Matrix&
{
    m_mode = Dense;
    return m_dense;
}

auto HessianBlock::dense() const -> const Matrix&
{
    return m_dense;
}

auto HessianBlock::eigenvalues() -> Vector&
{
    m_mode = EigenDecomposition;
    return m_eigenvalues;
}

auto HessianBlock::eigenvalues() const -> const Vector&
{
    return m_eigenvalues;
}

auto HessianBlock::eigenvectors() -> Matrix&
{
    m_mode = EigenDecomposition;
    return m_eigenvectors;
}

auto HessianBlock::eigenvectors() const -> const Matrix&
{
    return m_eigenvectors;
}

auto HessianBlock::eigenvectorsinv() -> Matrix&
{
    m_mode = EigenDecomposition;
    return m_eigenvectorsinv;
}

auto HessianBlock::eigenvectorsinv() const -> const Matrix&
{
    return m_eigenvectorsinv;
}

auto HessianBlock::mode() const -> Mode
{
    return m_mode;
}

auto HessianBlock::rows() const -> Index
{
    switch(mode())
    {
    case Zero: return m_dim;
    case Diagonal: return m_diagonal.rows();
    case Dense: return m_dense.rows();
    case EigenDecomposition: return m_eigenvalues.rows();
    }
}

auto HessianBlock::columns() const -> Index;

auto HessianBlock::coeff(Index i, Index j) const -> Scalar;

auto HessianBlock::operator()(Index i, Index j) const -> Scalar { return coeff(i, j); }

operator HessianBlock::PlainObject() const
{

}

} // namespace Optima
