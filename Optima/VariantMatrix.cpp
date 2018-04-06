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

#include "VariantMatrix.hpp"

namespace Optima {

VariantMatrix::VariantMatrix()
: structure(MatrixStructure::Zero)
{}

VariantMatrix::VariantMatrix(VariantMatrixConstRef other)
: dense(other.dense), diagonal(other.diagonal), structure(other.structure)
{}

auto VariantMatrix::operator=(MatrixConstRef mat) -> VariantMatrix&
{
    dense = mat;
    structure = MatrixStructure::Dense;
    return *this;
}

auto VariantMatrix::operator=(VectorConstRef vec) -> VariantMatrix&
{
    diagonal = vec;
    structure = MatrixStructure::Diagonal;
    return *this;
}

auto VariantMatrix::setZero() -> void
{
    structure = MatrixStructure::Zero;
}

auto VariantMatrix::setDense(Index size) -> void
{
    structure = size ? MatrixStructure::Dense : MatrixStructure::Zero;
    dense.resize(size, size);
}

auto VariantMatrix::setDiagonal(Index size) -> void
{
    structure = size ? MatrixStructure::Diagonal : MatrixStructure::Zero;
    diagonal.resize(size);
}

VariantMatrix::operator MatrixConstRef() const
{
    return dense;
}

VariantMatrix::operator VectorConstRef() const
{
    return diagonal;
}


VariantMatrixRef::VariantMatrixRef(VariantMatrix& mat)
: dense(mat.dense), diagonal(mat.diagonal), structure(mat.structure)
{}

VariantMatrixConstRef::VariantMatrixConstRef()
: dense(Matrix()), diagonal(Vector()), structure(MatrixStructure::Zero)
{}

VariantMatrixConstRef::VariantMatrixConstRef(VectorConstRef diagonal)
: dense(Matrix()), diagonal(diagonal), structure(matrixStructure(diagonal))
{}

VariantMatrixConstRef::VariantMatrixConstRef(const Vector& diagonal)
: VariantMatrixConstRef(VectorConstRef(diagonal))
{}

auto VariantMatrixConstRef::diagonalRef() -> VectorConstRef
{
    switch(structure) {
    case MatrixStructure::Diagonal: return diagonal;
    default: return dense.diagonal();
    }
}

VariantMatrixConstRef::VariantMatrixConstRef(MatrixConstRef dense)
: dense(dense), diagonal(Vector()), structure(matrixStructure(dense))
{}

VariantMatrixConstRef::VariantMatrixConstRef(const Matrix& dense)
: VariantMatrixConstRef(MatrixConstRef(dense))
{}

VariantMatrixConstRef::VariantMatrixConstRef(VariantMatrixRef mat)
: dense(mat.dense), diagonal(mat.diagonal), structure(mat.structure)
{}

VariantMatrixConstRef::VariantMatrixConstRef(const VariantMatrix& mat)
: dense(mat.dense), diagonal(mat.diagonal), structure(mat.structure)
{}

auto operator<<(MatrixRef mat, VariantMatrixConstRef vmat) -> MatrixRef
{
    switch(vmat.structure) {
    case MatrixStructure::Dense: mat = vmat.dense; break;
    case MatrixStructure::Diagonal: mat = diag(vmat.diagonal); break;
    case MatrixStructure::Zero: break;
    }
    return mat;
}

} // namespace Optima

