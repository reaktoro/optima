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
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "SaddlePointMatrix.hpp"

namespace Optima {

auto SaddlePointMatrix::matrix() const -> Matrix
{
    const Index n = H.rows();
    const Index m = A.rows();
    const Index t = n + m;
    Matrix res = zeros(t, t);
    res.topLeftCorner(n, n).diagonal() = H;
    res.topRightCorner(n, m)           = tr(A);
    res.bottomLeftCorner(m, n)         = A;
    rows(res, fixed).fill(0.0);
    for(Index i : fixed)
        res(i, i) = 1.0;
    return res;
}

auto SaddlePointVector::vector() const -> Vector
{
    const Index n = x.rows();
    const Index m = y.rows();
    const Index t = n + m;
    Vector res(t);
    res << x, y;
    return res;
}

auto operator*(const SaddlePointMatrix& mat, const SaddlePointVector& vec) -> SaddlePointVector
{
    const Matrix A = mat.matrix();
    const Vector x = vec.vector();
    const Vector b = A*x;
    SaddlePointVector res;
    res.x = b.head(mat.A.cols());
    res.y = b.tail(mat.A.rows());
    return res;
}

} // namespace Optima
