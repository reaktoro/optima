// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2016 Allan Leal
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

#include "doctest/doctest.hpp"

// Optima includes
#include <Optima/Optima.hpp>
using namespace Optima;

auto testSaddlePointMatrix() -> bool
{
    SaddlePointMatrix matrix;

    matrix.H = {1, 2, 3};
    matrix.A = {{1, 2, 3}, {3, 4, 5}};
    matrix.X = {3, 2, 1};
    matrix.Z = {5, 6, 7};

    const auto n = matrix.H.rows();
    const auto m = matrix.A.rows();
    const auto t = matrix.rows();
    Matrix M = zeros(t, t);
    M.topLeftCorner(n, n) = diag(matrix.H);
    M.topRightCorner(n, n) = -identity(n, n);
    M.middleRows(n, m).leftCols(n) = matrix.A;
    M.middleCols(n, m).topRows(n) = -tr(matrix.A);
    M.bottomLeftCorner(n, n) = diag(matrix.Z);
    M.bottomRightCorner(n, n) = diag(matrix.X);

    return matrix.isApprox(M);
}

TEST_CASE("Testing SaddlePointMatrix...")
{
	CHECK(testSaddlePointMatrix());
}



