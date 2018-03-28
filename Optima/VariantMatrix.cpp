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

#include "VariantMatrix.hpp"

namespace Optima {

VariantMatrix::VariantMatrix()
: _structure(MatrixStructure::Zero)
{}

auto VariantMatrix::structure() const -> MatrixStructure
{
    return _structure;
}

auto VariantMatrix::setZero() -> void
{
    _structure = MatrixStructure::Zero;
}

auto VariantMatrix::setDense(Index size) -> void
{
    _structure = size ? MatrixStructure::Dense : MatrixStructure::Zero;
    dense.resize(size, size);
}

auto VariantMatrix::setDiagonal(Index size) -> void
{
    _structure = size ? MatrixStructure::Diagonal : MatrixStructure::Zero;
    diagonal.resize(size);
}

VariantMatrixRef::VariantMatrixRef(VariantMatrix& mat)
: dense(mat.dense), diagonal(mat.diagonal), _structure(mat.structure())
{}

auto VariantMatrixRef::structure() const -> MatrixStructure
{
    return _structure;
}

VariantMatrixConstRef::VariantMatrixConstRef()
: dense(Matrix()), diagonal(Vector()), _structure(MatrixStructure::Zero)
{}

VariantMatrixConstRef::VariantMatrixConstRef(VectorConstRef diagonal)
: dense(Matrix()), diagonal(diagonal), _structure(matrixStructure(diagonal))
{}

VariantMatrixConstRef::VariantMatrixConstRef(const Vector& diagonal)
: VariantMatrixConstRef(VectorConstRef(diagonal))
{}

auto VariantMatrixConstRef::structure() const -> MatrixStructure
{
    return _structure;
}

auto VariantMatrixConstRef::topLeftCorner(Index size) const -> std::tuple<VariantMatrixConstRef, decltype(Eigen::seqN(0, size))>
{
    return std::make_tuple(*this, Eigen::seqN(0, size));
}

auto VariantMatrixConstRef::diagonalRef() -> VectorConstRef
{
    switch(_structure) {
    case MatrixStructure::Diagonal: return diagonal;
    default: return dense.diagonal();
    }
}

VariantMatrixConstRef::VariantMatrixConstRef(MatrixConstRef dense)
: dense(dense), diagonal(Vector()), _structure(matrixStructure(dense))
{}

VariantMatrixConstRef::VariantMatrixConstRef(const Matrix& dense)
: VariantMatrixConstRef(MatrixConstRef(dense))
{}

VariantMatrixConstRef::VariantMatrixConstRef(VariantMatrixRef mat)
: dense(mat.dense), diagonal(mat.diagonal), _structure(mat.structure())
{}

VariantMatrixConstRef::VariantMatrixConstRef(const VariantMatrix& mat)
: dense(mat.dense), diagonal(mat.diagonal), _structure(mat.structure())
{}

auto operator<<(MatrixRef mat, VariantMatrixConstRef vmat) -> MatrixRef
{
    switch(vmat.structure()) {
    case MatrixStructure::Dense: mat = vmat.dense; break;
    case MatrixStructure::Diagonal: mat = diag(vmat.diagonal); break;
    case MatrixStructure::Zero: break;
    }
    return mat;
}

} // namespace Optima

