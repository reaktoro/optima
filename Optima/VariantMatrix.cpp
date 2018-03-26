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

auto VariantMatrix::setZero() -> void
{
    _structure = MatrixStructure::Zero;
}

auto VariantMatrix::setDense(Index size) -> void
{
    _structure = MatrixStructure::Dense;
    _dense.resize(size, size);
}

auto VariantMatrix::setDiagonal(Index size) -> void
{
    _structure = MatrixStructure::Diagonal;
    _diagonal.resize(size);
}

auto VariantMatrix::structure() const -> MatrixStructure
{
    return _structure;
}

auto VariantMatrix::dense() const -> MatrixConstRef
{
    return _dense;
}

auto VariantMatrix::dense() -> MatrixRef
{
    return _dense;
}

auto VariantMatrix::diagonal() const -> VectorConstRef
{
    return _diagonal;
}

auto VariantMatrix::diagonal() -> VectorRef
{
    return _diagonal;
}


VariantMatrixRef::VariantMatrixRef(VariantMatrix& mat)
: dense(mat.dense()), diagonal(mat.diagonal()), structure(mat.structure())
{}


VariantMatrixConstRef::VariantMatrixConstRef()
: dense(Matrix()), diagonal(Vector()), structure(MatrixStructure::Zero)
{}

VariantMatrixConstRef::VariantMatrixConstRef(VectorConstRef diagonal)
: dense(Matrix()), diagonal(diagonal), structure(MatrixStructure::Diagonal)
{}

VariantMatrixConstRef::VariantMatrixConstRef(const Vector& diagonal)
: VariantMatrixConstRef(VectorConstRef(diagonal))
{}

VariantMatrixConstRef::VariantMatrixConstRef(MatrixConstRef dense)
: dense(dense), diagonal(Vector()), structure(MatrixStructure::Dense)
{}

VariantMatrixConstRef::VariantMatrixConstRef(const Matrix& dense)
: VariantMatrixConstRef(MatrixConstRef(dense))
{}

VariantMatrixConstRef::VariantMatrixConstRef(VariantMatrixRef mat)
: dense(mat.dense), diagonal(mat.diagonal), structure(mat.structure)
{}

VariantMatrixConstRef::VariantMatrixConstRef(const VariantMatrix& mat)
: dense(mat.dense()), diagonal(mat.diagonal()), structure(mat.structure())
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

