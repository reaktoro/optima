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

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

// Forward declarations
class HessianMatrix;
class HessianBlock;

} // namespace Optima

namespace Eigen {
namespace internal {

template<>
struct traits<Optima::HessianMatrix>
{
    typedef Eigen::Dense StorageKind;
    typedef Eigen::MatrixXpr XprKind;
    typedef Optima::Matrix::Scalar Scalar;
    typedef Optima::Matrix::Index Index;
    typedef Optima::Matrix::PlainObject PlainObject;
    enum {
        Flags = Eigen::ColMajor,
        RowsAtCompileTime = Optima::Matrix::RowsAtCompileTime,
        ColsAtCompileTime = Optima::Matrix::ColsAtCompileTime,
        MaxRowsAtCompileTime = Optima::Matrix::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = Optima::Matrix::MaxColsAtCompileTime,
        CoeffReadCost = Optima::Matrix::CoeffReadCost
    };
};

template<>
struct traits<Optima::HessianBlock>
{
    typedef Eigen::Dense StorageKind;
    typedef Eigen::MatrixXpr XprKind;
    typedef Optima::Matrix::Scalar Scalar;
    typedef Optima::Matrix::Index Index;
    typedef Optima::Matrix::PlainObject PlainObject;
    enum {
        Flags = Eigen::ColMajor,
        RowsAtCompileTime = Optima::Matrix::RowsAtCompileTime,
        ColsAtCompileTime = Optima::Matrix::ColsAtCompileTime,
        MaxRowsAtCompileTime = Optima::Matrix::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = Optima::Matrix::MaxColsAtCompileTime,
        CoeffReadCost = Optima::Matrix::CoeffReadCost
    };
};

} // namespace internal
} // namespace Eigen

namespace Optima {

/// Used to represent a square block matrix, \eq{H_i}, on the diagonal of the Hessian matrix, \eq{H}.
class HessianBlock : public Eigen::MatrixBase<HessianBlock>
{
public:
    /// Used to represent the possible modes for a Hessian block matrix.
    enum Mode
    {
        Zero, Diagonal, Dense, EigenDecomp
    };

    /// Used to represent an eigen decomposition of a Hessian block matrix.
    /// The eigen decomposition of a Hessian block matrix, \eq{H_i}, is a special decomposition
    /// denoted by \eq{H_i = V_i \Lambda_i V_{i}^{-1}}, where \eq{V_i} is the matrix whose columns
    /// correspond to the *eigenvectors* of \eq{H_i}, and \eq{\Lambda_i} is a diagonal matrix whose
    /// diagonal indices correspond to the *eigenvalues* of \eq{H_i}.
    struct EigenDecomposition
    {
        /// The matrix \eq{V_i} in the eigen decomposition.
        Matrix eigenvectors;

        /// The matrix \eq{\Lambda_i} in the eigen decomposition.
        Vector eigenvalues;

        /// The matrix \eq{V_{i}^{-1}} in the eigen decomposition.
        Matrix eigenvectorsinv;

        /// Return the value of the original matrix at (i, j).
        auto coeff(Index i, Index j) const -> double;

        /// Convert this EigenDecomposition instance into a Matrix instance.
        operator Matrix() const;
    };

    EIGEN_DENSE_PUBLIC_INTERFACE(HessianBlock)

    /// Construct a default HessianBlock instance.
    HessianBlock();

    /// Destroy this HessianBlock instance.
    virtual ~HessianBlock();

    /// Set the Hessian block matrix to a zero matrix mode.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::Zero.
    /// @param dim The dimension of the Hessian block matrix.
    auto zero(Index dim) -> void;

    /// Return a reference to the diagonal entries of a diagonal Hessian block matrix.
    /// This method also sets the Hessian block matrix to a diagonal matrix mode.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::Diagonal.
    /// @param dim The dimension of the Hessian block matrix.
    auto diagonal(Index dim) -> Vector&;

    /// Return a const reference to the diagonal entries of a diagonal Hessian block matrix.
    auto diagonal() const -> const Vector&;

    /// Return a reference to the matrix representing the dense Hessian block matrix.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::Dense.
    /// @param dim The dimension of the Hessian block matrix.
    auto dense(Index dim) -> Matrix&;

    /// Return a const reference to the matrix representing the dense Hessian block matrix.
    auto dense() const -> const Matrix&;

    /// Return a reference to the eigendecomposition of this Hessian block matrix.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::EigenDecomp.
    /// @param dim The dimension of the Hessian block matrix.
    auto eigendecomposition(Index dim) -> EigenDecomposition&;

    /// Return a const reference to the eigendecomposition of this Hessian block matrix.
    auto eigendecomposition() const -> const EigenDecomposition&;

    /// Return the mode of the Hessian matrix.
    auto mode() const -> Mode;

    /// Return the number of rows in the Hessian block matrix.
    auto rows() const -> Index;

    /// Return the number of columns in the Hessian block matrix.
    auto cols() const -> Index;

    /// Return an entry of the Hessian block matrix.
    auto coeff(Index i, Index j) const -> Scalar;

    /// Return an entry of the block diagonal matrix.
    auto operator()(Index i, Index j) const -> Scalar { return coeff(i, j); }

    /// Convert this HessianBlock instance to a Matrix instance.
    operator PlainObject() const;

private:
    /// The dimension of this square Hessian block matrix (only used when zero mode is active).
    Index m_dim;

    /// The current mode of this Hessian block matrix.
    Mode m_mode;

    /// The diagonal vector of this diagonal Hessian block matrix (if diagonal mode active).
    Vector m_diagonal;

    /// The matrix representing a dense Hessian block matrix (if dense mode active).
    Matrix m_dense;

    /// The diagonal matrix \eq{\Lambda} in the eigendecomposition of this Hessian block matrix (if eigendecomposition mode active).
    EigenDecomposition m_eigen;
};

/// Used to represent a Hessian matrix in various forms.
class HessianMatrix : public Eigen::MatrixBase<HessianMatrix>
{
public:
    EIGEN_DENSE_PUBLIC_INTERFACE(HessianMatrix)

    /// Alias to nested HessianBlock type EigenDecomposition.
    using EigenDecomposition = HessianBlock::EigenDecomposition;

    /// Construct a default HessianMatrix instance.
    HessianMatrix();

    /// Destroy this HessianMatrix instance.
    virtual ~HessianMatrix();

    /// Set the Hessian matrix to a zero square matrix.
    /// @param dim The dimension of the Hessian matrix.
    auto zero(Index dim) -> void;

    /// Return a reference to the diagonal entries of the diagonal Hessian matrix.
    /// @param dim The dimension of the Hessian matrix.
    auto diagonal(Index dim) -> Vector&;

    /// Return a const reference to the diagonal entries of the diagonal Hessian matrix.
    auto diagonal() const -> const Vector&;

    /// Return a reference to the dense Hessian matrix.
    /// @param dim The dimension of the Hessian matrix.
    auto dense(Index dim) -> Matrix&;

    /// Return a const reference to the dense Hessian matrix.
    auto dense() const -> const Matrix&;

    /// Return a reference to the eigendecomposition of this Hessian block matrix.
    /// @param dim The dimension of the Hessian matrix.
    auto eigendecomposition(Index dim) -> EigenDecomposition&;

    /// Return a const reference to the eigendecomposition of this Hessian block matrix.
    auto eigendecomposition() const -> const EigenDecomposition&;

    /// Return a reference to a block matrix on the diagonal of the Hessian matrix.
    /// @param iblock The index of the block matrix.
    auto block(Index iblock) -> HessianBlock&;

    /// Return a const reference to a block matrix on the diagonal of the Hessian matrix.
    /// @param iblock The index of the block matrix.
    auto block(Index iblock) const -> const HessianBlock&;

    /// Return a const reference to the Hessian block matrices on the diagonal of the Hessian matrix.
    /// @param numblocks The number of Hessian block matrices.
    auto blocks(Index numblocks) -> std::vector<HessianBlock>&;

    /// Return a const reference to the Hessian block matrices on the diagonal of the Hessian matrix.
    auto blocks() const -> const std::vector<HessianBlock>&;

    /// Return the number of rows of the Hessian matrix.
    auto rows() const -> Index;

    /// Return the number of columns of the Hessian matrix.
    auto cols() const -> Index;

    /// Return an entry of the block diagonal matrix.
    auto coeff(Index i, Index j) const -> Scalar;

    /// Return an entry of the Hessian matrix.
    auto operator()(Index i, Index j) const -> Scalar { return coeff(i, j); }

    /// Convert this HessianMatrix instance to a Matrix instance.
    operator PlainObject() const;

private:
    /// The Hessian block matrices on the diagonal of the Hessian matrix.
    /// Use just one block for full dense Hessian matrices.
    std::vector<HessianBlock> m_blocks;
};

} // namespace Optima
