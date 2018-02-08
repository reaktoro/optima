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

#include <catch.hpp>

// Optima includes
#include <Optima/Core/IpSaddlePointMatrix.hpp>
using namespace Eigen;
using namespace Optima;

TEST_CASE("Testing IpSaddlePointMatrix...")
{
    MatrixXd H = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    MatrixXd A = {{1, 2, 3}, {3, 4, 5}};
    VectorXd Z = {1, 2, 3};
    VectorXd W = {4, 5, 6};
    VectorXd L = {9, 8, 7};
    VectorXd U = {6, 5, 4};
    Index n = 3;
    Index m = 2;
    Index nx = 3;
    Index nf = 0;

    IpSaddlePointMatrix mat(H, A, Z, W, L, U, nx, nf);

    MatrixXd M = {
        {1,  2,  3,  1,  3, -1,  0,  0, -1,  0,  0},
        {4,  5,  6,  2,  4,  0, -1,  0,  0, -1,  0},
        {7,  8,  9,  3,  5,  0,  0, -1,  0,  0, -1},
        {1,  2,  3,  0,  0,  0,  0,  0,  0,  0,  0},
        {3,  4,  5,  0,  0,  0,  0,  0,  0,  0,  0},
        {1,  0,  0,  0,  0,  9,  0,  0,  0,  0,  0},
        {0,  2,  0,  0,  0,  0,  8,  0,  0,  0,  0},
        {0,  0,  3,  0,  0,  0,  0,  7,  0,  0,  0},
        {4,  0,  0,  0,  0,  0,  0,  0,  6,  0,  0},
        {0,  5,  0,  0,  0,  0,  0,  0,  0,  5,  0},
        {0,  0,  6,  0,  0,  0,  0,  0,  0,  0,  4},
    };

    // Check conversion to a Matrix instance
    REQUIRE(M.isApprox(mat.matrix()));

    SECTION("Testing conversion when some variables are fixed")
    {
        nx = 2;
        nf = 1;

        IpSaddlePointMatrix mat(H, A, Z, W, L, U, nx, nf);

        M.middleRows(nx, nf).fill(0.0);
        M.middleCols(nx, nf).topRows(n).fill(0.0);
        M.block(nx, nx, nf, nf) = identity(nf, nf);
        M.middleRows(n + m + nx, nf).middleCols(nx, nf).fill(0.0);
        M.bottomRows(nf).middleCols(nx, nf).fill(0.0);
        M.bottomRightCorner(2*n, 2*n).diagonal().segment(nx, nf).fill(1.0);
        M.bottomRightCorner(2*n, 2*n).diagonal().tail(nf).fill(1.0);

        // Check conversion to a Matrix instance
        REQUIRE(M.isApprox(mat.matrix()));
    }
}

TEST_CASE("Testing IpSaddlePointVector...")
{
    Index n = 5;
    Index m = 3;
    Index t = 3*n + m;

    VectorXd r = linspace(t, 1, t);

    IpSaddlePointVector vec(r, n, m);

    REQUIRE(r.head(n).isApprox(vec.a()));
    REQUIRE(r.segment(n, m).isApprox(vec.b()));
    REQUIRE(r.segment(n + m, n).isApprox(vec.c()));
    REQUIRE(r.tail(n).isApprox(vec.d()));
}
