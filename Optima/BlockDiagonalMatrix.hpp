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
// on with this program. If not, see <http://www.gnu.org/licenses/>.

#pragma once

// C++ includes
#include <vector>

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

// Forward declarations
class BlockDiagonalMatrix;

} // namespace Optima

namespace Eigen {
namespace internal {

template<>
struct traits<Optima::BlockDiagonalMatrix>
{
    typedef Eigen::Dense StorageKind;
    typedef Eigen::MatrixXpr XprKind;
    typedef Optima::MatrixXd::StorageIndex StorageIndex;
    typedef Optima::MatrixXd::Scalar Scalar;
    enum {
        Flags = Eigen::ColMajor,
        RowsAtCompileTime = Optima::MatrixXd::RowsAtCompileTime,
        ColsAtCompileTime = Optima::MatrixXd::ColsAtCompileTime,
        MaxRowsAtCompileTime = Optima::MatrixXd::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = Optima::MatrixXd::MaxColsAtCompileTime,
    };
};

} // namespace internal
} // namespace Eigen

namespace Optima {

/// Used to represent a block diagonal matrix.
class BlockDiagonalMatrix : public Eigen::MatrixBase<BlockDiagonalMatrix>
{
public:
    EIGEN_DENSE_PUBLIC_INTERFACE(BlockDiagonalMatrix)

    /// Construct a default BlockDiagonalMatrix instance.
    BlockDiagonalMatrix() {}

    /// Construct a default BlockDiagonalMatrix instance.
    /// @param m The number of rows in the block diagonal matrix.
    /// @param n The number of columns in the block diagonal matrix.
    BlockDiagonalMatrix(Index numblocks)
    : m_blocks(numblocks) {}

    /// Destroy this BlockDiagonalMatrix instance.
    virtual ~BlockDiagonalMatrix() {}

    /// Return a reference to a block matrix on the diagonal.
    /// @param i The index of the block matrix.
    auto block(Index i) -> MatrixXd& { return m_blocks[i]; }

    /// Return a const reference to a block matrix on the diagonal.
    /// @param i The index of the block matrix.
    auto block(Index i) const -> const MatrixXd& { return m_blocks[i]; }

    /// Return a reference to the block matrices on the diagonal.
    auto blocks() -> std::vector<MatrixXd>& { return m_blocks; }

    /// Return a const reference to the block matrices on the diagonal.
    auto blocks() const -> const std::vector<MatrixXd>& { return m_blocks; }

    /// Return the number of rows of the block diagonal matrix.
    auto rows() const -> Index
    {
        Index sum = 0;
        for(const MatrixXd& block : blocks())
            sum += block.rows();
        return sum;
    }

    /// Return the number of columns of the block diagonal matrix.
    auto cols() const -> Index
    {
        Index sum = 0;
        for(const MatrixXd& block : blocks())
            sum += block.cols();
        return sum;
    }

    // Delete this resize overload method.
    auto resize(Index rows, Index cols) -> void = delete;

    /// Resize the block diagonal matrix.
    /// @param numblocks The number of blocks on the block diagonal matrix.
    auto resize(Index numblocks) -> void
    {
        m_blocks.resize(numblocks);
    }

    /// Return an entry of the block diagonal matrix.
    auto coeff(Index i, Index j) const -> Scalar
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

    /// Convert this BlockDiagonalMatrix instance to a Matrix instance.
    operator PlainObject() const
    {
        const Index m = rows();
        const Index n = cols();
        const Index nblocks = blocks().size();
        PlainObject res = PlainObject::Zero(m, n);
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

private:
    /// The block matrices on the block diagonal matrix.
    std::vector<MatrixXd> m_blocks;
};

} // namespace Optima

namespace Eigen {
namespace internal {

template<>
struct evaluator<Optima::BlockDiagonalMatrix> : evaluator_base<Optima::BlockDiagonalMatrix>
{
    typedef Optima::BlockDiagonalMatrix XprType;
    typedef Optima::MatrixXd::Scalar Scalar;
    enum
    {
        CoeffReadCost = evaluator<Optima::MatrixXd>::CoeffReadCost,
        Flags = Eigen::ColMajor
    };

    evaluator(const Optima::BlockDiagonalMatrix& view)
    : m_mat(view) {}

    auto coeff(Index row, Index col) const -> Scalar { return m_mat.coeff(row, col); }

    const Optima::BlockDiagonalMatrix& m_mat;
};

} // namespace internal
} // namespace Eigen
