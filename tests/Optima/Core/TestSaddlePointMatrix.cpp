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

TEST_CASE("Testing SaddlePointMatrix...")
{
    SaddlePointMatrix mat;

    mat.H = {1, 2, 3};
    mat.A = {{1, 2, 3}, {3, 4, 5}};

    Matrix M = {
		{1,  0,  0, -1, -3, -1,  0,  0},
		{0,  2,  0, -2, -4,  0, -1,  0},
		{0,  0,  3, -3, -5,  0,  0, -1},
		{1,  2,  3,  0,  0,  0,  0,  0},
		{3,  4,  5,  0,  0,  0,  0,  0},
		{5,  0,  0,  0,  0,  3,  0,  0},
		{0,  6,  0,  0,  0,  0,  2,  0},
		{0,  0,  7,  0,  0,  0,  0,  1}};

	SUBCASE("checking conversion to a Matrix instance")
	{
		// Check conversion to a Matrix instance
		CHECK(M.isApprox(mat.matrix()));
	}
}

TEST_CASE("Testing SaddlePointVector...")
{
    SaddlePointVector vec;

    vec.x = {1, 2, 3};
    vec.y = {6, 7};

    Vector V = {1, 2, 3, 6, 7, 3, 2, 1};

	SUBCASE("checking conversion to a Vector instance")
	{
		// Check conversion to a Vector instance
		CHECK(V.isApprox(vec.vector()));
	}
}
