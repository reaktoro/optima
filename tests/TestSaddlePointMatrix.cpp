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

#include <doctest/doctest.hpp>

// Optima includes
#include <Optima/Optima.hpp>
using namespace Optima;

TEST_CASE("Testing SaddlePointMatrix...")
{
    SaddlePointMatrix matrix;

    matrix.H = {1, 2, 3};
    matrix.A = {{1, 2, 3}, {3, 4, 5}};
    matrix.X = {3, 2, 1};
    matrix.Z = {5, 6, 7};

    REQUIRE(matrix.valid());

    REQUIRE(matrix.rows() == 8);
    REQUIRE(matrix.cols() == 8);

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
		// Check `coeff` method implementation
		CHECK(M.isApprox(matrix));

		// Check conversion to a Matrix instance
		CHECK(M.isApprox(Matrix(matrix)));
	}

	SUBCASE("checking conversion when both X and Z are empty")
	{
		matrix.X = matrix.Z = {};

		REQUIRE(matrix.valid());

		REQUIRE(matrix.rows() == 5);
		REQUIRE(matrix.cols() == 5);

		M.conservativeResize(5, 5);

		// Check `coeff` method implementation
		CHECK(M.isApprox(matrix));

		// Check conversion to a Matrix instance
		CHECK(M.isApprox(Matrix(matrix)));
	}

	SUBCASE("checking valid() when X and Z have incompatible dimensions")
	{
		matrix.X = {1}; matrix.Z = {};
		CHECK_FALSE(matrix.valid());
	}

	SUBCASE("checking valid() when H and A have incompatible dimensions")
	{
		matrix.H = {1, 2};
		CHECK_FALSE(matrix.valid());
	}

	SUBCASE("checking valid() when H and X have incompatible dimensions")
	{
		matrix.H = {1, 2};
		CHECK_FALSE(matrix.valid());
	}
}

TEST_CASE("Testing SaddlePointVector...")
{
    SaddlePointVector vector;

    vector.a = {1, 2, 3};
    vector.b = {6, 7};
    vector.c = {3, 2, 1};

    REQUIRE(vector.valid());

    REQUIRE(vector.rows() == 8);

    Vector V = {1, 2, 3, 6, 7, 3, 2, 1};

	SUBCASE("checking conversion to a Vector instance")
	{
		// Check `coeff` method implementation
		CHECK(V.isApprox(vector));

		// Check conversion to a Matrix instance
		CHECK(V.isApprox(Vector(vector)));
	}

	SUBCASE("checking conversion when c is empty")
	{
		vector.c = {};

		REQUIRE(vector.valid());

		REQUIRE(vector.rows() == 5);

		V.conservativeResize(5);

		// Check `coeff` method implementation
		CHECK(V.isApprox(vector));

		// Check conversion to a Matrix instance
		CHECK(V.isApprox(Vector(vector)));
	}

	SUBCASE("checking valid() when a and c have incompatible dimensions")
	{
		vector.c = {1, 2};
		CHECK_FALSE(vector.valid());
	}
}

TEST_CASE("Testing SaddlePointMatrixCanonical...")
{
    SaddlePointMatrixCanonical matrix;

    matrix.Gb = {1, 2, 3};
    matrix.Gs = {4, 5};
    matrix.Gu = {6};

    matrix.Bb = {9, 8, 7};
    matrix.Bs = {{1, 2}, {2, 3}, {3, 4}};
    matrix.Bu = {5, 6, 7};

    matrix.Eb = {1, 2, 3};
    matrix.Es = {4, 5};
    matrix.Eu = {6};

    Matrix M = {
		{1, 0, 0, 0, 0, 0, 9, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 2, 0, 0, 0, 0, 0, 8, 0, 0, 2, 0, 0, 0, 0},
		{0, 0, 3, 0, 0, 0, 0, 0, 7, 0, 0, 3, 0, 0, 0},
		{0, 0, 0, 4, 0, 0, 1, 2, 3, 0, 0, 0, 4, 0, 0},
		{0, 0, 0, 0, 5, 0, 2, 3, 4, 0, 0, 0, 0, 5, 0},
		{0, 0, 0, 0, 0, 6, 5, 6, 7, 0, 0, 0, 0, 0, 6},
		{9, 0, 0, 1, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 8, 0, 2, 3, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{0, 0, 7, 3, 4, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		{1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
		{0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0},
		{0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0},
		{0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0},
		{0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0},
		{0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 0, 0, 0, 6}};

    REQUIRE(matrix.valid());

	CHECK(matrix.rows() == 15);
	CHECK(matrix.cols() == 15);

	SUBCASE("checking conversion to a Matrix instance")
	{
		// Check `coeff` method implementation
		REQUIRE(M.isApprox(matrix));

		// Check conversion to a Matrix instance
		CHECK(M.isApprox(Matrix(matrix)));
	}

	SUBCASE("checking conversion when E is empty")
	{
		matrix.Eb = matrix.Es = matrix.Eu = {};

		REQUIRE(matrix.valid());

		CHECK(matrix.rows() == 9);
		CHECK(matrix.cols() == 9);

		M.conservativeResize(9, 9);

		// Check `coeff` method implementation
		CHECK(M.isApprox(matrix));

		// Check conversion to a Matrix instance
		CHECK(M.isApprox(Matrix(matrix)));
	}

	SUBCASE("checking convertion when Gu, Bu, Eu are empty")
	{
		matrix.Gu = matrix.Bu = matrix.Eu = {};

		REQUIRE(matrix.valid());

	    Matrix M = {
	        {1, 0, 0, 0, 0, 9, 0, 0, 1, 0, 0, 0, 0},
	        {0, 2, 0, 0, 0, 0, 8, 0, 0, 2, 0, 0, 0},
	        {0, 0, 3, 0, 0, 0, 0, 7, 0, 0, 3, 0, 0},
	        {0, 0, 0, 4, 0, 1, 2, 3, 0, 0, 0, 4, 0},
	        {0, 0, 0, 0, 5, 2, 3, 4, 0, 0, 0, 0, 5},
	        {9, 0, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0},
	        {0, 8, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0},
	        {0, 0, 7, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0},
	        {1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
	        {0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0},
	        {0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0},
	        {0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 4, 0},
	        {0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 5}};

		// Check `coeff` method implementation
		CHECK(M.isApprox(matrix));

		// Check conversion to a Matrix instance
		CHECK(M.isApprox(Matrix(matrix)));
	}

	SUBCASE("checking convertion when Gs, Gu, Bs, Bu, Es, Eu are empty")
	{
		matrix.Gs = matrix.Bs = matrix.Es = {};
		matrix.Gu = matrix.Bu = matrix.Eu = {};

		REQUIRE(matrix.valid());

		Matrix M = {
		    {1, 0, 0, 9, 0, 0, 1, 0, 0},
		    {0, 2, 0, 0, 8, 0, 0, 2, 0},
		    {0, 0, 3, 0, 0, 7, 0, 0, 3},
		    {9, 0, 0, 0, 0, 0, 0, 0, 0},
		    {0, 8, 0, 0, 0, 0, 0, 0, 0},
		    {0, 0, 7, 0, 0, 0, 0, 0, 0},
		    {1, 0, 0, 0, 0, 0, 1, 0, 0},
		    {0, 2, 0, 0, 0, 0, 0, 2, 0},
		    {0, 0, 3, 0, 0, 0, 0, 0, 3}};

		// Check `coeff` method implementation
		CHECK(M.isApprox(matrix));

		// Check conversion to a Matrix instance
		CHECK(M.isApprox(Matrix(matrix)));
	}
}

TEST_CASE("Testing SaddlePointVectorCanonical...")
{
    SaddlePointVectorCanonical vector;

    vector.ab = {1, 2, 3};
    vector.as = {4, 5};
    vector.au = {6};
    vector.b  = {7, 8};
    vector.cb = {9, 8, 7};
    vector.cs = {6, 5};
    vector.cu = {4};

    REQUIRE(vector.valid());

    REQUIRE(vector.rows() == 14);

    Vector V = {1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4};

	SUBCASE("checking conversion to a Vector instance")
	{
		// Check `coeff` method implementation
		CHECK(V.isApprox(vector));

		// Check conversion to a Matrix instance
		CHECK(V.isApprox(Vector(vector)));
	}

	SUBCASE("checking conversion when au and cu are empty")
	{
		vector.au = vector.cu = {};

		REQUIRE(vector.valid());

		REQUIRE(vector.rows() == 12);

		Vector V = {1, 2, 3, 4, 5, 7, 8, 9, 8, 7, 6, 5};

		// Check `coeff` method implementation
		CHECK(V.isApprox(vector));

		// Check conversion to a Matrix instance
		CHECK(V.isApprox(Vector(vector)));
	}

	SUBCASE("checking conversion when as, au, cs and cu are empty")
	{
		vector.as = vector.cs = {};
		vector.au = vector.cu = {};

		REQUIRE(vector.valid());

		REQUIRE(vector.rows() == 8);

		Vector V = {1, 2, 3, 7, 8, 9, 8, 7};

		// Check `coeff` method implementation
		CHECK(V.isApprox(vector));

		// Check conversion to a Matrix instance
		CHECK(V.isApprox(Vector(vector)));
	}

	SUBCASE("checking valid() when ab and cb have incompatible dimensions")
	{
		vector.ab = {};
		CHECK_FALSE(vector.valid());
	}

	SUBCASE("checking valid() when as and cs have incompatible dimensions")
	{
		vector.as = {};
		CHECK_FALSE(vector.valid());
	}

	SUBCASE("checking valid() when au and cu have incompatible dimensions")
	{
		vector.au = {};
		CHECK_FALSE(vector.valid());
	}
}
