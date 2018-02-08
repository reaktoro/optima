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
// on with this program. If not, see <http://www.gnu.org/licenses/>.

#pragma once

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// Used to represent a square block matrix, \eq{H_i}, on the diagonal of the Hessian matrix, \eq{H}.
class HessianBlock
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

        /// Convert this EigenDecomposition instance into a Matrix instance.
        auto convert() const -> Matrix;
    };

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

    /// Return the dimension of the Hessian block matrix.
    auto dim() const -> Index;

    /// Convert this HessianBlock instance to a Matrix instance.
    auto convert() const -> Matrix;

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
class HessianMatrix
{
public:
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

    /// Return the dimension of the Hessian matrix.
    auto dim() const -> Index;

    /// Convert this HessianMatrix instance to a Matrix instance.
    auto convert() const -> Matrix;

private:
    /// The Hessian block matrices on the diagonal of the Hessian matrix.
    /// Use just one block for full dense Hessian matrices.
    std::vector<HessianBlock> m_blocks;
};

} // namespace Optima
