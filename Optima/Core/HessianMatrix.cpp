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

auto HessianBlock::EigenDecomposition::coeff(Index i, Index j) const -> double
{
    const auto& V = eigenvectors;
    const auto& L = eigenvalues;
    const auto& Vinv = eigenvectorsinv;
    return V.row(i) * diag(L) * Vinv.col(j);
}

HessianBlock::EigenDecomposition::operator Matrix() const
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
    m_diagonal.resize(dim, dim);
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

auto HessianBlock::rows() const -> Index
{
    return m_dim;
}

auto HessianBlock::cols() const -> Index
{
    return m_dim;
}

auto HessianBlock::coeff(Index i, Index j) const -> Scalar
{
    switch(m_mode)
    {
    case Diagonal: return i == j ? m_diagonal[i] : 0.0;
    case Dense: return m_dense(i, j);
    case EigenDecomp: return m_eigen.coeff(i, j);
    default: return 0.0;
    }
}

HessianBlock::operator PlainObject() const
{
    switch(m_mode)
    {
    case Diagonal: return diag(m_diagonal);
    case Dense: return m_dense;
    case EigenDecomp: return Matrix(m_eigen);
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

auto HessianMatrix::rows() const -> Index
{
    Index sum = 0;
    for(const auto& block : m_blocks)
        sum += block.rows();
    return sum;
}

auto HessianMatrix::cols() const -> Index
{
    return rows();
}

auto HessianMatrix::coeff(Index i, Index j) const -> Scalar
{
    eigen_assert(i < rows() && j < cols());
    const Index nblocks = blocks().size();
    Index irow = 0, icol = 0;
    for(Index iblock = 0; iblock < nblocks; ++iblock)
    {
        const Index nrows = block(iblock).rows();
        const Index ncols = block(iblock).cols();
        if(i < irow + nrows)
        {
            if(icol <= j && j < icol + ncols)
                return block(iblock)(i - irow, j - icol);
            else return 0.0;
        }
        irow += nrows;
        icol += ncols;
    }
    return 0.0;
}

HessianMatrix::operator PlainObject() const
{
    Index dim = rows();
    const Index nblocks = blocks().size();
    PlainObject res = zeros(dim, dim);
    Index irow = 0, icol = 0;
    for(Index i = 0; i < nblocks; ++i)
    {
        const Index nrows = block(i).rows();
        const Index ncols = block(i).cols();
        res.block(irow, icol, nrows, ncols) = block(i);
        irow += nrows;
        icol += ncols;
    }
    return res;
}

} // namespace Optima
