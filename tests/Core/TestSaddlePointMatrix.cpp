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

#include <doctest/doctest.hpp>

// Optima includes
#include <Optima/Core/HessianMatrix.hpp>
#include <Optima/Core/SaddlePointMatrix.hpp>
using namespace Eigen;
using namespace Optima;

TEST_CASE("Testing SaddlePointMatrix...")
{
    MatrixXd H = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    MatrixXd A = {{1, 2, 3}, {3, 4, 5}};

    SaddlePointMatrix mat(H, A);

    MatrixXd M = {
		{1,  2,  3, 1, 3},
		{4,  5,  6, 2, 4},
		{7,  8,  9, 3, 5},
		{1,  2,  3, 0, 0},
		{3,  4,  5, 0, 0}
    };

    // Check conversion to a Matrix instance
    CHECK(M.isApprox(mat.matrix()));

    SUBCASE("Testing conversion when some variables are fixed")
    {
        Indices fixed = {1};

        SaddlePointMatrix mat(H, A, fixed);

        M.row(1).fill(0.0);
        M(1, 1) = 1.0;

        // Check conversion to a Matrix instance
        CHECK(M.isApprox(mat.matrix()));
    }
}

TEST_CASE("Testing SaddlePointVector...")
{
    Index n = 5;
    Index m = 3;
    Index t = n + m;

    VectorXd r = linspace(t, 1, t);

    SaddlePointVector vec(r, n, m);

    CHECK(r.head(n).isApprox(vec.a()));
    CHECK(r.tail(m).isApprox(vec.b()));
}
