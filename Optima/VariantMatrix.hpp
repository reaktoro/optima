// Optima is a C++ library for solving linear and non-linear constrained optimization problems
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

// // Forward declarations
// class VariantMatrixRef;
// class VariantMatrixConstRef;

// /// Used to represent a matrix that can be either dense, diagonal, or zero.
// class VariantMatrix
// {
// public:
//     /// The entries of the variant matrix with dense structure.
//     Matrix dense;

//     /// The entries of the variant matrix with diagonal structure.
//     Vector diagonal;

//     /// The current structure of the variant matrix.
//     MatrixStructure structure;

//     /// Construct a default VariantMatrix instance that represents a zero matrix.
//     VariantMatrix();

//     /// Construct a VariantMatrix instance from a given VariantMatrixConstRef.
//     VariantMatrix(VariantMatrixConstRef other);

//     /// Assign a matrix to this VariantMatrix instance, setting its structure to dense matrix.
//     auto operator=(MatrixConstRef dense) -> VariantMatrix&;

//     /// Assign a vector to this VariantMatrix instance, setting its structure to diagonal matrix.
//     auto operator=(VectorConstRef diagonal) -> VariantMatrix&;

//     /// Set the matrix structure to zero.
//     auto setZero() -> void;

//     /// Set the matrix structure to dense and resize @ref dense with given size.
//     /// @param size The size of the dense matrix.
//     auto setDense(Index size) -> void;

//     /// Set the matrix structure to diagonal and resize @ref diagonal with given size.
//     /// @param size The size of the diagonal matrix.
//     auto setDiagonal(Index size) -> void;

//     /// Convert this VariantMatrix instance into a constant reference to the dense matrix.
//     operator MatrixConstRef() const;

//     /// Convert this VariantMatrix instance into a constant reference to the diagonal matrix.
//     operator VectorConstRef() const;
// };

// /// Used to represent a reference to a variant matrix.
// /// @see VariantMatrix
// class VariantMatrixRef
// {
// public:
//     /// The entries of the variant matrix with dense structure.
//     MatrixRef dense;

//     /// The entries of the variant matrix with diagonal structure.
//     VectorRef diagonal;

//     /// The current structure of the variant matrix.
//     const MatrixStructure structure;

//     /// Construct a VariantMatrixRef instance from a variant matrix.
//     VariantMatrixRef(VariantMatrix& mat);
// };

// /// Used to represent a constant reference to a variant matrix.
// /// @see VariantMatrix
// class VariantMatrixConstRef
// {
// public:
//     /// The entries of the variant matrix with dense structure.
//     MatrixConstRef dense;

//     /// The entries of the variant matrix with diagonal structure.
//     VectorConstRef diagonal;

//     /// The current structure of the variant matrix.
//     const MatrixStructure structure;

//     /// Construct a VariantMatrixConstRef instance with zero structure.
//     VariantMatrixConstRef();

//     /// Construct a VariantMatrixConstRef instance with given dense matrix.
//     VariantMatrixConstRef(MatrixConstRef dense);
//     VariantMatrixConstRef(const Matrix& dense);

//     /// Construct a VariantMatrixConstRef instance with given diagonal matrix.
//     VariantMatrixConstRef(VectorConstRef diagonal);
//     VariantMatrixConstRef(const Vector& diagonal);

//     /// Construct a VariantMatrixConstRef instance from a variant matrix.
//     VariantMatrixConstRef(VariantMatrixRef mat);
//     VariantMatrixConstRef(const VariantMatrix& mat);

//     /// Return an indexed view of the variant matrix.
//     template<typename IndicesType>
//     auto operator()(const IndicesType& indices) const -> decltype(std::make_tuple(*this, indices))
//     {
//         return std::make_tuple(*this, indices);
//     }

//     /// Return a view to the top left corner of the variant matrix.
//     auto topLeftCorner(Index size) const -> decltype(std::make_tuple(*this, Eigen::seqN(0, size)))
//     {
//         return std::make_tuple(*this, Eigen::seqN(0, size));
//     }

//     /// Return a reference to the diagonal entries of the variant matrix.
//     auto diagonalRef() -> VectorConstRef;
// };

// /// Assign a VariantMatrixBase object to a Matrix instance.
// auto operator<<(MatrixRef mat, VariantMatrixConstRef vmat) -> MatrixRef;

// /// Assign an indexed view of a VariantMatrixBase object to a Matrix instance.
// template<typename IndicesType>
// auto operator<<(MatrixRef mat, const std::tuple<VariantMatrixConstRef, IndicesType>& view) -> MatrixRef
// {
//     const auto& hessian = std::get<0>(view);
//     const auto& indices = std::get<1>(view);
//     switch(hessian.structure) {
//     case MatrixStructure::Dense: mat = hessian.dense(indices, indices); break;
//     case MatrixStructure::Diagonal: mat = diag(hessian.diagonal(indices)); break;
//     case MatrixStructure::Zero: break;
//     }
//     return mat;
// }

} // namespace Optima

