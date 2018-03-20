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
    Eigen::Ref<MatrixType> dense;

    /// The Hessian matrix as a diagonal matrix.
    Eigen::Ref<VectorType> diagonal;

    /// The structure of the Hessian matrix.
    const MatrixStructure structure;

    /// Construct a default HessianMatrixBase instance that represents a zero Hessian matrix.
    /// @param dense The Hessian matrix as a dense matrix.
    HessianMatrixBase()
    : dense(Matrix()), diagonal(Vector()), structure(MatrixStructure::Diagonal)
    {}

    /// Construct a HessianMatrixBase instance with given dense Hessian matrix.
    /// @param dense The Hessian matrix as a dense matrix.
    HessianMatrixBase(MatrixType& dense)
    : dense(dense), diagonal(Vector()), structure(MatrixStructure::Diagonal) {}

    /// Construct a HessianMatrixBase instance with given dense Hessian matrix.
    /// @param dense The Hessian matrix as a dense matrix.
    HessianMatrixBase(Eigen::Ref<MatrixType> matview)
    : dense(matview.cols() > 1 ? matview : Eigen::Ref<MatrixType>(Matrix())),
      diagonal(matview.cols() > 1 ? Vector() : matview.col(0)),
      structure(matview.cols() > 1 ? MatrixStructure::Dense : MatrixStructure::Diagonal) {}

    /// Construct a HessianMatrixBase instance with given diagonal Hessian matrix.
    /// @param diagonal The Hessian matrix as a diagonal matrix, represented by a vector.
    HessianMatrixBase(VectorType& diagonal)
    : dense(Matrix()), diagonal(diagonal), structure(MatrixStructure::Diagonal) {}

    /// Construct a HessianMatrixBase instance with given diagonal Hessian matrix.
    /// @param diagonal The Hessian matrix as a diagonal matrix, represented by a vector.
    HessianMatrixBase(Eigen::Ref<VectorType> diagonal)
    : dense(Matrix()), diagonal(diagonal), structure(MatrixStructure::Diagonal) {}

    /// Construct a HessianMatrixBase instance with given dense and diagonal Hessian matrices.
    /// @param dense The Hessian matrix as a dense matrix.
    /// @param diagonal The Hessian matrix as a diagonal matrix.
    HessianMatrixBase(MatrixType& dense, VectorType& diagonal)
    : dense(dense), diagonal(diagonal), structure(dense.size() ? MatrixStructure::Dense : MatrixStructure::Diagonal) {}

    /// Construct a HessianMatrixBase instance with given dense and diagonal Hessian matrices.
    /// @param dense The Hessian matrix as a dense matrix.
    /// @param diagonal The Hessian matrix as a diagonal matrix.
    HessianMatrixBase(Eigen::Ref<MatrixType> dense, Eigen::Ref<VectorType> diagonal)
    : dense(dense), diagonal(diagonal), structure(dense.size() ? MatrixStructure::Dense : MatrixStructure::Diagonal) {}

    /// Construct a HessianMatrixBase instance from another.
    template<typename MatrixOther, typename VectorOther>
    HessianMatrixBase(HessianMatrixBase<MatrixOther, VectorOther> other)
    : dense(other.dense), diagonal(other.diagonal), structure(other.structure) {}

    /// Return a reference to the diagonal entries of the Hessian matrix.
    auto diagonalRef() -> Eigen::Ref<VectorType>
    {
        switch(structure) {
        case MatrixStructure::Diagonal: return diagonal;
        default: return dense.diagonal();
        }
    }

    /// Return an indexed view of the Hessian matrix.
    template<typename IndicesType>
    auto operator()(const IndicesType& indices) const -> decltype(std::make_tuple(*this, indices))
    {
        return std::make_tuple(*this, indices);
    }
};

/// Used to represent a reference to a Hessian matrix.
using HessianMatrixRef = HessianMatrixBase<Matrix, Vector>;

/// Used to represent a constant reference to a Hessian matrix.
using HessianMatrixConstRef = HessianMatrixBase<const Matrix, const Vector>;

/// Assign a HessianMatrixBase object to a Matrix instance.
template<typename MatrixType, typename VectorType>
auto operator<<(MatrixRef mat, const HessianMatrixBase<MatrixType, VectorType>& hessian) -> MatrixRef
{
    switch(hessian.structure) {
    case MatrixStructure::Dense: mat = hessian.dense; break;
    case MatrixStructure::Diagonal: return mat.diagonal() = hessian.diagonal; break;
    case MatrixStructure::Zero: break;
    }
    return mat;
}

/// Assign an indexed view of a HessianMatrixBase object to a Matrix instance.
template<typename HessianMatrixType, typename IndicesType>
auto operator<<(MatrixRef mat, const std::tuple<HessianMatrixType, IndicesType>& view) -> MatrixRef
{
    const auto& hessian = std::get<0>(view);
    const auto& indices = std::get<1>(view);
    switch(hessian.structure) {
    case MatrixStructure::Dense: mat = hessian.dense(indices, indices); break;
    case MatrixStructure::Diagonal: mat.diagonal() = hessian.diagonal(indices); break;
    case MatrixStructure::Zero: break;
    }
    return mat;
}

} // namespace Optima

