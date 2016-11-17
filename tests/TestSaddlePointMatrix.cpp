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

auto testSaddlePointMatrix() -> void
{
    SaddlePointMatrix matrix;

    matrix.H = {1, 2, 3};
    matrix.A = {{1, 2, 3}, {3, 4, 5}};
    matrix.X = {3, 2, 1};
    matrix.Z = {5, 6, 7};

    REQUIRE(matrix.valid());

	CHECK(matrix.rows() == 8);
	CHECK(matrix.cols() == 8);

    SUBCASE("checking conversion to a Matrix instance")
    {
		const auto n = matrix.H.rows();
		const auto m = matrix.A.rows();
		const auto t = matrix.rows();
		Matrix M = zeros(t, t);
		M.topLeftCorner(n, n).diagonal() = matrix.H;
		M.topRightCorner(n, n).diagonal() = -ones(n);
		M.middleRows(n, m).leftCols(n) = matrix.A;
		M.middleCols(n, m).topRows(n) = -tr(matrix.A);
		M.bottomLeftCorner(n, n).diagonal() = matrix.Z;
		M.bottomRightCorner(n, n).diagonal() = matrix.X;

		// Check `coeff` method implementation
		CHECK(M.isApprox(matrix));

		// Check conversion to a Matrix instance
		CHECK(M.isApprox(Matrix(matrix)));

		SUBCASE("checking conversion when both X and Z are empty")
		{
			matrix.X = matrix.Z = {};

			REQUIRE(matrix.valid());

			CHECK(matrix.rows() == 5);
			CHECK(matrix.cols() == 5);

			M.conservativeResize(5, 5);

			// Check `coeff` method implementation
			CHECK(M.isApprox(matrix));

			// Check conversion to a Matrix instance
			CHECK(M.isApprox(Matrix(matrix)));
		}
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

//auto testSaddlePointMatrixCanonical() -> bool
//{
//    SaddlePointMatrixCanonical matrix;
//
//    matrix.Gb = {1, 2, 3};
//    matrix.Gs = {4, 5};
//    matrix.Gu = {6};
//
//    matrix.Bb = {9, 8, 7};
//    matrix.Bs = {{1, 2}, {2, 3}, {3, 4}};
//    matrix.Bu = {5, 6, 7};
//
//    matrix.Eb = {1, 2, 3};
//    matrix.Es = {4, 5};
//    matrix.Eu = {6};
//
//    Matrix M(matrix);
//
//    std::cout << matrix << std::endl;
//    std::cout << "M = \n" << M << std::endl;
//
//    return true;
//}

TEST_CASE("Testing SaddlePointMatrix...")
{
	testSaddlePointMatrix();
//	testSaddlePointMatrixCanonical();
//	CHECK(testSaddlePointMatrix());
//	CHECK(testSaddlePointMatrixCanonical());
}



