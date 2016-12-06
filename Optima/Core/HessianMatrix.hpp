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
        Zero, Diagonal, Dense, EigenDecomposition
    };

    EIGEN_DENSE_PUBLIC_INTERFACE(HessianBlock)

    /// Construct a default HessianBlock instance.
    HessianBlock();

    /// Destroy this HessianBlock instance.
    virtual ~HessianBlock();

    /// Set the Hessian block matrix to a zero matrix mode.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::Zero.
    /// @param dim The dimension of the square zero matrix.
    auto zero(Index dim) -> void;

    /// Return a reference to the diagonal entries of a diagonal Hessian block matrix.
    /// This method also sets the Hessian block matrix to a diagonal matrix mode.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::Diagonal.
    auto diagonal() -> Vector&;

    /// Return a const reference to the diagonal entries of a diagonal Hessian block matrix.
    auto diagonal() const -> const Vector&;

    /// Return a reference to the matrix representing the dense Hessian block matrix.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::Dense.
    auto dense() -> Matrix&;

    /// Return a const reference to the matrix representing the dense Hessian block matrix.
    auto dense() const -> const Matrix&;

    /// Return a reference to the eigenvalues of this Hessian block matrix.
    /// This method returns a reference to the entries in the diagonal matrix \eq{\Lambda_i}.
    /// These are the eigenvalues of the Hessian block matrix \eq{H_i}, whose
    /// eigendecomposition is denoted by \eq{H_i = V_i \Lambda_i V_{i}^{-1}}, with \eq{V_i}
    /// denoting the matrix whose columns are the eigenvectors of \eq{H_i}.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::EigenDecomposition.
    auto eigenvalues() -> Vector&;

    /// Return a const reference to the eigenvalues of this Hessian block matrix.
    /// This method returns a const reference to the entries in the diagonal matrix \eq{\Lambda_i}.
    /// These are the eigenvalues of the Hessian block matrix \eq{H_i}, whose
    /// eigendecomposition is denoted by \eq{H_i = V_i \Lambda_i V_{i}^{-1}}, with \eq{V_i}
    /// denoting the matrix whose columns are the eigenvectors of \eq{H_i}.
    auto eigenvalues() const -> const Vector&;

    /// Return a reference to the eigenvectors of this Hessian block matrix.
    /// This method returns a reference to the matrix \eq{V_i}.
    /// This is the matrix whose columns are the eigenvectors of this Hessian block matrix, \eq{H_i},
    /// which is used to represent the eigendecomposition \eq{H_i = V_i \Lambda_i V_{i}^{-1}},
    /// where \eq{\Lambda_i} denotes a diagonal matrix with the eigenvalues of \eq{H_i}.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::EigenDecomposition.
    auto eigenvectors() -> Matrix&;

    /// Return a const reference to the eigenvectors of this Hessian block matrix.
    /// This method returns a const reference to the matrix \eq{V_i}.
    /// This is the matrix whose columns are the eigenvectors of this Hessian block matrix, \eq{H_i},
    /// which is used to represent the eigendecomposition \eq{H_i = V_i \Lambda_i V_{i}^{-1}},
    /// where \eq{\Lambda_i} denotes a diagonal matrix with the eigenvalues of \eq{H_i}.
    auto eigenvectors() const -> const Matrix&;

    /// Return a reference to the inverse of the eigenvectors of this Hessian block matrix.
    /// This method returns a const reference to the matrix \eq{V_{i}^{-1}}, the inverse of the
    /// matrix of eigenvectors, \eq{V_i}, of this Hessian block matrix.
    /// @note After calling this method, a call to method @ref mode will return HessianBlock::EigenDecomposition.
    auto eigenvectorsinv() -> Matrix&;

    /// Return a const reference to the inverse of the eigenvectors of this Hessian block matrix.
    /// This method returns a const reference to the matrix \eq{V_{i}^{-1}}, the inverse of the
    /// matrix of eigenvectors, \eq{V_i}, of this Hessian block matrix.
    auto eigenvectorsinv() const -> const Matrix&;

    /// Return the mode of the Hessian matrix.
    auto mode() const -> Mode;

    /// Return the number of rows in the Hessian block matrix.
    auto rows() const -> Index;

    /// Return the number of columns in the Hessian block matrix.
    auto columns() const -> Index;

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
    Vector m_eigenvalues;

    /// The matrix \eq{V} in the eigendecomposition of this Hessian block matrix (if eigendecomposition mode active).
    Vector m_eigenvectors;

    /// The matrix \eq{V^-1} in the eigendecomposition of this Hessian block matrix (if eigendecomposition mode active).
    Vector m_eigenvectorsinv;
};

/// Used to represent a Hessian matrix in various forms.
class HessianMatrix : public Eigen::MatrixBase<HessianMatrix>
{
public:
    EIGEN_DENSE_PUBLIC_INTERFACE(HessianMatrix)

    /// Construct a default HessianMatrix instance.
    HessianMatrix();

    /// Destroy this HessianMatrix instance.
    virtual ~HessianMatrix();

    /// Set the Hessian matrix to a zero matrix form.
    /// Use this method if the Hessian is a zero matrix, such as those found in
    /// linear programming problems.
    /// ~~~{.cpp}
    /// using namespace Optima; {delete}
    /// HessianMatrix H;
    /// H.setModeZero(10); // set the Hessian matrix to a 10x10 zero matrix
    /// ~~~
    /// @param dim The dimension of the Hessian matrix.
    /// @see setModeDiagonal, setModeDense, setModeBlockDiagonal, setModeEigenDecomposition
    auto setModeZero(Index dim) -> void;

    /// Set the Hessian matrix to diagonal form.
    /// Use this method if the Hessian matrix is a diagonal matrix, such as those found in
    /// quadratic programming problems.
    /// Use method @ref diagonal to get a reference to the diagonal vector of the Hessian matrix.
    /// ~~~{.cpp}
    /// using namespace Optima; {delete}
    /// HessianMatrix H;
    /// H.setModeDiagonal(10);     // set the Hessian matrix as a 10x10 diagonal matrix
    /// H.diagonal() = random(10); // set the diagonal entries of the Hessian matrix to random values.
    /// ~~~
    /// @param dim The dimension of the Hessian matrix.
    /// @see setModeZero, setModeDense, setModeBlockDiagonal, setModeEigenDecomposition
    auto setModeDiagonal(Index dim) -> void;

    /// Set the Hessian matrix to dense form.
    /// Use this method if the Hessian matrix is a symmetric dense matrix with no special structure.
    /// Use method @ref dense to get a reference to the matrix representing the Hessian matrix.
    /// ~~~{.cpp}
    /// using namespace Optima; {delete}
    /// HessianMatrix H;
    /// H.setModeDense(3);   // set the Hessian matrix as a 3x3 dense matrix
    /// H.dense() = {        // set the Hessian matrix to the given 3x3 symmetric matrix
    ///     {1.0, 2.0, 3.0},
    ///     {2.0, 4.0, 6.0},
    ///     {3.0, 6.0, 8.0}
    /// };
    /// ~~~
    /// @param dim The dimension of the Hessian matrix.
    /// @see setModeZero, setModeDiagonal, setModeBlockDiagonal, setModeEigenDecomposition
    auto setModeDense(Index dim) -> void;

    /// Set the Hessian matrix to block diagonal form.
    /// Use this method if the Hessian matrix is a block diagonal matrix, where each block matrix
    /// on its diagonal is symmetric.
    /// Use method @ref block to get a reference to the block matrix with given index.
    /// ~~~{.cpp}
    /// using namespace Optima; {delete}
    /// HessianMatrix H;
    /// H.setModeBlockDiagonal(3); // set the Hessian matrix as a block diagonal matrix with 3 blocks
    /// H.block(0) = random(2, 2); // set the block 0 to a 2x2 random matrix
    /// H.block(1) = random(5, 5); // set the block 1 to a 5x5 random matrix
    /// H.block(2) = random(7, 7); // set the block 2 to a 7x7 random matrix
    /// ~~~
    /// @param numblocks The number of blocks on the diagonal of the Hessian matrix.
    /// @see setModeZero, setModeDiagonal, setModeDense, setModeEigenDecomposition
    auto setModeBlockDiagonal(Index numblocks) -> void;

    /// Set the Hessian matrix to eigendecomposition form.
    /// Use this method if the Hessian matrix has a known [eigendecomposition](http://mathworld.wolfram.com/EigenDecomposition.html),
    /// in which the Hessian matrix \eq{H} can be decomposed as \eq{H = V \Lambda V^-1}, where
    /// \eq{V} is the matrix whose columns are the *eigenvectors* of \eq{H} and \eq{\Lambda} is a
    /// diagonal matrix with the *eigenvalues* of \eq{H}.
    /// Use method @ref eigenvalues, @ref eigenvectors, and @ref eigenvectorsinv to get a reference
    /// to the matrices \eq{\Lambda}, \eq{V}, and \eq{V^-1}, respectively.
    /// @see setModeZero, setModeDiagonal, setModeDense, setModeBlockDiagonal
    auto setModeEigenDecomposition() -> void;

    /// Return a reference to the dense Hessian matrix.
    /// @note This method should be used together with @ref setModeDense.
    auto dense() -> Matrix&;

    /// Return a const reference to the dense Hessian matrix.
    /// @note This method should be used together with @ref setModeDense.
    auto dense() const -> const Matrix&;

    /// Return a reference to the diagonal entries of the diagonal Hessian matrix.
    /// @note This method should be used together with @ref setModeDiagonal.
    auto diagonal() -> Vector&;

    /// Return a const reference to the diagonal entries of the diagonal Hessian matrix.
    /// @note This method should be used together with @ref setModeDiagonal.
    auto diagonal() const -> const Vector&;

    /// Return a reference to a block matrix on the diagonal of the Hessian matrix.
    /// @note This method should be used together with @ref setModeBlockDiagonal.
    /// @param iblock The index of the block matrix.
    auto block(Index iblock) -> Matrix&;

    /// Return a const reference to a block matrix on the diagonal of the Hessian matrix.
    /// @note This method should be used together with @ref setModeBlockDiagonal.
    /// @param iblock The index of the block matrix.
    auto block(Index iblock) const -> const Matrix&;

    /// Return a reference to the block matrices on the diagonal of the Hessian matrix.
    /// @note This method should be used together with @ref setModeBlockDiagonal.
    auto blocks() -> std::vector<Matrix>&;

    /// Return a const reference to the block matrices on the diagonal of the Hessian matrix.
    /// @note This method should be used together with @ref setModeBlockDiagonal.
    auto blocks() const -> const std::vector<Matrix>&;

    /// Return a reference to the diagonal entries of matrix \eq{\Lambda} in the eigendecomposition \eq{H = V \Lambda V^-1}.
    /// @note This method should be used together with @ref setModeEigenDecomposition.
    auto eigenvalues() -> Vector&;

    /// Return a const reference to the diagonal entries of matrix \eq{\Lambda} in the eigendecomposition \eq{H = V \Lambda V^-1}.
    /// @note This method should be used together with @ref setModeEigenDecomposition.
    auto eigenvalues() const -> const Vector&;

    /// Return a reference to the matrix \eq{V} in the eigendecomposition \eq{H = V \Lambda V^-1}.
    /// @note This method should be used together with @ref setModeEigenDecomposition.
    auto eigenvectors() -> Matrix&;

    /// Return a const reference to the matrix \eq{V} in the eigendecomposition \eq{H = V \Lambda V^-1}.
    /// @note This method should be used together with @ref setModeEigenDecomposition.
    auto eigenvectors() const -> const Matrix&;

    /// Return a reference to the matrix \eq{V^-1} in the eigendecomposition \eq{H = V \Lambda V^-1}.
    /// @note This method should be used together with @ref setModeEigenDecomposition.
    auto eigenvectorsinv() -> Matrix&;

    /// Return a const reference to the matrix \eq{V^-1} in the eigendecomposition \eq{H = V \Lambda V^-1}.
    /// @note This method should be used together with @ref setModeEigenDecomposition.
    auto eigenvectorsinv() const -> const Matrix&;

    /// Return the mode of the Hessian matrix.
    auto mode() const -> Mode;

    /// Return the number of rows of the Hessian matrix.
    auto rows() const -> Index;

    /// Return the number of columns of the Hessian matrix.
    auto cols() const -> Index;

private:
    Mode m_mode;

    Vector m_diagonal;

    BlockDiagonalMatrix m_blockdiagonal;

    std::vector<Vector> m_eigenvalues;

    std::vector<Matrix> m_eigenvectors;

    std::vector<Matrix> m_eigenvectorsinv;
};

} // namespace Optima
