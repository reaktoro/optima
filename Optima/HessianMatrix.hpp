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

#pragma once

// Optima includes
#include <Optima/Matrix.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

/// Used to represent a Hessian matrix, either dense, diagonal, or zero matrix.
template<typename MatrixType, typename VectorType>
class HessianMatrixBase
{
public:
    /// The Hessian matrix as a dense matrix.
    MatrixType dense;

    /// The Hessian matrix as a diagonal matrix.
    VectorType diagonal;

    /// The structure of the Hessian matrix.
    const MatrixStructure structure;

    /// Construct a default HessianMatrixBase instance that represents a zero Hessian matrix.
    /// @param dense The Hessian matrix as a dense matrix.
    HessianMatrixBase()
    : dense(Matrix::Zero(0, 0)), diagonal(Matrix::Zero(0)), structure(Zero) {}

    /// Construct a HessianMatrixBase instance with given dense Hessian matrix.
    /// @param dense The Hessian matrix as a dense matrix.
    HessianMatrixBase(MatrixType&& dense)
    : dense(dense), diagonal(Matrix::Zero(0)), structure(Dense) {}

    /// Construct a HessianMatrixBase instance with given diagonal Hessian matrix.
    /// @param diagonal The Hessian matrix as a diagonal matrix, represented by a vector.
    HessianMatrixBase(VectorType&& diagonal)
    : dense(Matrix::Zero(0, 0)), diagonal(diagonal), structure(Diagonal) {}

    /// Construct a HessianMatrixBase instance from another.
    template<typename MatrixOther, typename VectorOther>
    HessianMatrixBase(HessianMatrixBase<MatrixOther, VectorOther> other)
    : dense(other.dense), diagonal(other.diagonal), structure(other.structure) {}

    /// Return a reference to the diagonal entries of the Hessian matrix.
    auto diagonalRef() -> VectorRef { return dense.size() ? dense.diagonal() : diagonal; }

    /// Return a const reference to the diagonal entries of the Hessian matrix.
    auto diagonalRef() const -> VectorConstRef { return dense.size() ? dense.diagonal() : diagonal; }
};

/// Used to represend an indexed view of a Hessian matrix.
template<typename MatrixType, typename VectorType, typename RowIndices, typename ColIndices>
class HessianMatrixIndexedView
{
public:
    /// Construct a default HessianMatrixIndexedView instance.
    HessianMatrixIndexedView();


private:
};
/// Used to represent a reference to a Hessian matrix.
using HessianMatrixRef = HessianMatrixBase<MatrixRef, VectorRef>;

/// Used to represent a constant reference to a Hessian matrix.
using HessianMatrixConstRef = HessianMatrixBase<MatrixConstRef, VectorConstRef>;

/// Assign a HessianMatrixBase object to a Matrix instance.
template<typename MatrixType, typename VectorType>
auto operator<<(MatrixRef mat, const HessianMatrixBase<MatrixType, VectorType>& hessian) -> Matrix&
{
    switch(hessian.structure) {
    case Dense: mat = hessian.dense; break;
    case Diagonal: return mat.diagonal() = hessian.diagonal; break;
    case Zero: break;
    }
    return mat;
}

} // namespace Optima

