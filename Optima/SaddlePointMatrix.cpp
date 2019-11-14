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

#include "SaddlePointMatrix.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

SaddlePointMatrix::SaddlePointMatrix(MatrixConstRef H, VectorConstRef D, MatrixConstRef A, IndicesConstRef jf)
: SaddlePointMatrix(H, D, A, Matrix{}, jf)
{}

SaddlePointMatrix::SaddlePointMatrix(MatrixConstRef H, VectorConstRef D, MatrixConstRef A, MatrixConstRef G, IndicesConstRef jf)
: H(H), D(D), A(A), G(G), jf(jf)
{
    Assert(H.rows() == H.cols() || H.cols() == 1,
        "Could not create a SaddlePointMatrix object.",
            "Matrix H is neither a square n-by-n matrix or a diagonal n-by-1 matrix.");

    Assert(H.rows() == A.cols(),
        "Could not create a SaddlePointMatrix object.",
            "Matrix A must have the same number of columns as there are rows in H.");

    Assert(D.rows() == 0 || D.rows() == A.cols(),
        "Could not create a SaddlePointMatrix object.",
            "Matrix D must be an empty vector or have the same number of rows as there are rows in H.");

    Assert(A.rows() < A.cols(),
        "Could not create a SaddlePointMatrix object.",
            "Matrix A must have less number of rows than number of columns.");

    Assert(G.size() == 0 || G.rows() == A.rows(),
        "Could not create a SaddlePointMatrix object.",
            "Matrix G, when non-zero, must have the same number of rows and columns as there are rows in A.");
}

SaddlePointMatrix::operator Matrix() const
{
    const auto m = A.rows();
    const auto n = A.cols();
    const auto t = m + n;

    Matrix res = zeros(t, t);

    switch(matrixStructure(H)) {
    case MatrixStructure::Dense:
        res.topLeftCorner(n, n) = H;
        res.topLeftCorner(n, n)(jf, all).fill(0.0);
        res.topLeftCorner(n, n)(all, jf).fill(0.0);
        res.topLeftCorner(n, n).diagonal()(jf).fill(1.0);
        break;
    case MatrixStructure::Diagonal:
        res.topLeftCorner(n, n).diagonal() = H.col(0);
        break;
    case MatrixStructure::Zero:
        break;
    }

    // Add the D contribution if D is non-empty
    if(D.size()) res.topLeftCorner(n, n).diagonal() += D;

    // Set all entries in H + D block corresponding to fixed variables
    res.topLeftCorner(n, n).diagonal()(jf).fill(1.0);

    res.topRightCorner(n, m) = tr(A);
    res.topRightCorner(n, m)(jf, all).fill(0.0);

    res.bottomLeftCorner(m, n) = A;
    res.bottomRightCorner(m, m) << G;

    return res;
}

auto operator*(SaddlePointMatrix lhs, VectorConstRef rhs) -> Vector
{
    Matrix M(lhs);
    Vector res = M * rhs;
    return res;
}

SaddlePointVector::SaddlePointVector(VectorConstRef x, VectorConstRef y)
: x(x), y(y), a(x), b(y)
{}

SaddlePointVector::SaddlePointVector(VectorConstRef r, Index n, Index m)
: x(r.head(n)), y(r.tail(m)), a(x), b(y)
{}

template<typename SaddlePointVectorType>
auto toVector(const SaddlePointVectorType& vec) -> Vector
{
    const auto n = vec.x.rows();
    const auto m = vec.y.rows();
    const auto t = n + m;
    Vector res(t);
    res << vec.x, vec.y;
    return res;
}

SaddlePointVector::operator Vector() const
{
    return toVector(*this);
}

SaddlePointSolution::SaddlePointSolution(VectorRef x, VectorRef y)
: x(x), y(y)
{}

SaddlePointSolution::SaddlePointSolution(VectorRef r, Index n, Index m)
: x(r.head(n)), y(r.tail(m))
{}

auto SaddlePointSolution::operator=(VectorConstRef vec) -> SaddlePointSolution&
{
    x = vec.head(x.rows());
    y = vec.tail(y.rows());
    return *this;
}

SaddlePointSolution::operator Vector() const
{
    return toVector(*this);
}

} // namespace Optima
