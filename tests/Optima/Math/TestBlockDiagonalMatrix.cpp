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
#include <Optima/Optima.hpp>
using namespace Optima;

TEST_CASE("Testing BlockDiagonalMatrix - square matrix")
{
    const Index numblocks = 3;

	Matrix M = {
	    {1, 2, 3, 0, 0, 0},
	    {2, 3, 4, 0, 0, 0},
	    {3, 4, 5, 0, 0, 0},
	    {0, 0, 0, 1, 2, 0},
	    {0, 0, 0, 2, 3, 0},
	    {0, 0, 0, 0, 0, 6}
	};

	BlockDiagonalMatrix A(numblocks);
	A.block(0) = M.block(0, 0, 3, 3);
	A.block(1) = M.block(3, 3, 2, 2);
	A.block(2) = M.block(5, 5, 1, 1);

	CHECK(A.isApprox(M));
	CHECK(M.isApprox(Matrix(A)));
}

TEST_CASE("Testing BlockDiagonalMatrix - rectangular matrix m < n")
{
    const Index numblocks = 2;

	Matrix M = {
	    {1, 2, 3, 0, 0, 0},
	    {2, 3, 4, 0, 0, 0},
	    {3, 4, 5, 0, 0, 0},
	    {0, 0, 0, 1, 2, 0},
	    {0, 0, 0, 2, 3, 0}
	};

	BlockDiagonalMatrix A(numblocks);
	A.block(0) = M.block(0, 0, 3, 3);
	A.block(1) = M.block(3, 3, 2, 3);

	CHECK(A.isApprox(M));
	CHECK(M.isApprox(Matrix(A)));
}

TEST_CASE("Testing BlockDiagonalMatrix - rectangular matrix m < n")
{
    const Index numblocks = 2;

    Matrix M = {
        {1, 2, 3, 0, 0},
        {2, 3, 4, 0, 0},
        {3, 4, 5, 0, 0},
        {0, 0, 0, 1, 2},
        {0, 0, 0, 2, 3},
        {0, 0, 0, 0, 0}
    };

	BlockDiagonalMatrix A(numblocks);
	A.block(0) = M.block(0, 0, 3, 3);
	A.block(1) = M.block(3, 3, 3, 2);

	CHECK(A.isApprox(M));
	CHECK(M.isApprox(Matrix(A)));
}

