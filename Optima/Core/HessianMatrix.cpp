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

auto HessianBlock::EigenDecomposition::convert() const -> Matrix
{
    const auto& V = eigenvectors;
    const auto& L = eigenvalues;
    const auto& Vinv = eigenvectorsinv;
    return V * diag(L) * Vinv;
}

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

auto HessianBlock::diagonal(Index dim) -> Vector&
{
    m_dim = dim;
    m_mode = Diagonal;
    m_diagonal.resize(dim);
    return m_diagonal;
}

auto HessianBlock::diagonal() const -> const Vector&
{
    return m_diagonal;
}

auto HessianBlock::dense(Index dim) -> Matrix&
{
    m_dim = dim;
    m_mode = Dense;
    m_dense.resize(dim, dim);
    return m_dense;
}

auto HessianBlock::dense() const -> const Matrix&
{
    return m_dense;
}

auto HessianBlock::eigendecomposition(Index dim) -> EigenDecomposition&
{
    m_dim = dim;
    m_mode = EigenDecomp;
    m_eigen.eigenvalues.resize(dim);
    m_eigen.eigenvectors.resize(dim, dim);
    m_eigen.eigenvectorsinv.resize(dim, dim);
    return m_eigen;
}

auto HessianBlock::eigendecomposition() const -> const EigenDecomposition&
{
    return m_eigen;
}

auto HessianBlock::mode() const -> Mode
{
    return m_mode;
}

auto HessianBlock::dim() const -> Index
{
    return m_dim;
}

auto HessianBlock::convert() const -> Matrix
{
    switch(m_mode)
    {
    case Diagonal: return diag(m_diagonal);
    case Dense: return m_dense;
    case EigenDecomp: return m_eigen.convert();
    default: return zeros(m_dim, m_dim);
    }
}

HessianMatrix::HessianMatrix()
: m_blocks(1)
{}

HessianMatrix::~HessianMatrix()
{}

auto HessianMatrix::zero(Index dim) -> void
{
    m_blocks.resize(1);
    m_blocks.front().zero(dim);
}

auto HessianMatrix::diagonal(Index dim) -> Vector&
{
    m_blocks.resize(1);
    return m_blocks.front().diagonal(dim);
}

auto HessianMatrix::diagonal() const -> const Vector&
{
    return m_blocks.front().diagonal();
}

auto HessianMatrix::dense(Index dim) -> Matrix&
{
    m_blocks.resize(1);
    return m_blocks.front().dense(dim);
}

auto HessianMatrix::dense() const -> const Matrix&
{
    return m_blocks.front().dense();
}

auto HessianMatrix::eigendecomposition(Index dim) -> EigenDecomposition&
{
    m_blocks.resize(1);
    return m_blocks.front().eigendecomposition(dim);
}

auto HessianMatrix::eigendecomposition() const -> const EigenDecomposition&
{
    return m_blocks.front().eigendecomposition();
}

auto HessianMatrix::blocks(Index numblocks) -> std::vector<HessianBlock>&
{
    m_blocks.resize(numblocks);
    return m_blocks;
}

auto HessianMatrix::blocks() const -> const std::vector<HessianBlock>&
{
    return m_blocks;
}

auto HessianMatrix::block(Index iblock) -> HessianBlock&
{
    return m_blocks[iblock];
}

auto HessianMatrix::block(Index iblock) const -> const HessianBlock&
{
    return m_blocks[iblock];
}

auto HessianMatrix::dim() const -> Index
{
    Index sum = 0;
    for(const auto& block : m_blocks)
        sum += block.dim();
    return sum;
}

auto HessianMatrix::convert() const -> Matrix
{
    const Index nblocks = blocks().size();
    Matrix res = zeros(dim(), dim());
    Index irow = 0;
    for(Index i = 0; i < nblocks; ++i)
    {
        const Index bdim = block(i).dim();
        res.block(irow, irow, bdim, bdim) = block(i).convert();
        irow += bdim;
    }
    return res;
}

} // namespace Optima
