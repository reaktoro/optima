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
#include <Optima/Common/Index.hpp>
#include <Optima/Math/Canonicalizer.hpp>
using namespace Eigen;
using namespace Optima;

#define CHECK_CANONICAL_FORM                                     \
{                                                                \
    const auto& R = canonicalizer.R();                           \
    const auto& Q = canonicalizer.Q();                           \
    const auto& C = canonicalizer.C();                           \
    CHECK((R * A * Q - C).norm() == approx(0.0));                \
}                                                                \

#define CHECK_CANONICAL_ORDERING                                 \
{                                                                \
    const Index n = canonicalizer.numVariables();                \
    const Index r = canonicalizer.numBasicVariables();           \
    auto ibasic = canonicalizer.Q().indices().head(r);           \
    auto inonbasic = canonicalizer.Q().indices().tail(n - r);    \
    for(Index i = 1; i < ibasic.size(); ++i)                     \
        CHECK(w[ibasic[i]] <= w[ibasic[i - 1]]);                 \
    for(Index i = 1; i < inonbasic.size(); ++i)                  \
        CHECK(w[inonbasic[i]] <= w[inonbasic[i - 1]]);           \
}                                                                \

TEST_CASE("Testing Canonicalizer")
{
	const Index m = 4;
	const Index n = 10;

	MatrixXd A = random(m, n);

    Canonicalizer canonicalizer(A);

    const Index r = canonicalizer.numBasicVariables();

	CHECK_CANONICAL_FORM

	for(Index i = 0; i < r; ++i)
	{
		for(Index j = 0; j < n - r; ++j)
		{
			canonicalizer.swapBasicVariable(i, j);
			CHECK_CANONICAL_FORM
		}
	}
}

TEST_CASE("Testing Canonicalizer with two linearly dependent rows")
{
	const Index m = 4;
	const Index n = 10;

	MatrixXd A = random(m, n);
	A.row(2) = A.row(0) + 2*A.row(1);
	A.row(3) = A.row(1) - 2*A.row(2);

    Canonicalizer canonicalizer(A);

    const Index r = canonicalizer.numBasicVariables();

	CHECK_CANONICAL_FORM

	for(Index i = 0; i < r; ++i)
	{
		for(Index j = 0; j < n - r; ++j)
		{
			canonicalizer.swapBasicVariable(i, j);
			CHECK_CANONICAL_FORM
		}
	}
}

TEST_CASE("Testing the update method of the Canonicalizer class")
{

	const MatrixXd A = {
	//    H2O   H+    OH-   HCO3- CO2   CO3--
		{ 2,    1,    1,    1,    0,    0 }, // H
		{ 1,    0,    1,    3,    2,    3 }, // O
		{ 0,    0,    0,    1,    1,    1 }, // C
		{ 0,    1,   -1,   -1,    0,   -2 }  // Z
	};

	Canonicalizer canonicalizer(A);

    const Index r = canonicalizer.numBasicVariables();

	CHECK(r == 3);
	CHECK_CANONICAL_FORM

	VectorXd w = {55.1, 1.e-4, 1.e-10, 0.1, 0.5, 1e-2};

	canonicalizer.update(w);

	CHECK_CANONICAL_FORM
	CHECK_CANONICAL_ORDERING

	Eigen::VectorXi expectedQ = {0, 4, 3, 5, 1, 2};
	Eigen::VectorXi actualQ = canonicalizer.Q().indices();

	CHECK(expectedQ.isApprox(actualQ));

	w = {55.1, 1.e-4, 1.e-10, 0.3, 0.1, 0.8};

	canonicalizer.update(w);

	CHECK_CANONICAL_FORM
	CHECK_CANONICAL_ORDERING

	expectedQ = {0, 5, 3, 4, 1, 2};
	actualQ = canonicalizer.Q().indices();

	CHECK(expectedQ.isApprox(actualQ));
}

TEST_CASE("Testing the update method of the Canonicalizer class with fixed variables")
{
    const MatrixXd A = {
    //    H2O   H+    OH-   HCO3- CO2   CO3--
        { 2,    1,    1,    1,    0,    0 }, // H
        { 1,    0,    1,    3,    2,    3 }, // O
        { 0,    0,    0,    1,    1,    1 }, // C
        { 0,    1,   -1,   -1,    0,   -2 }  // Z
    };

    Canonicalizer canonicalizer(A);

    const Index r = canonicalizer.numBasicVariables();

    CHECK(r == 3);
    CHECK_CANONICAL_FORM

    Indices ifixed = {3, 4, 5};

    VectorXd w = {55.1, 1.e-4, 1.e-10, -0.1, -0.5, -1e-2};

    canonicalizer.update(w);

    CHECK_CANONICAL_FORM
    CHECK_CANONICAL_ORDERING
//    Eigen::VectorXi expectedQ = {0, 1, 2, 3, 4, 5};
//    Eigen::VectorXi actualQ = canonicalizer.Q().indices();
//
//    CHECK(expectedQ.isApprox(actualQ));
//
//    w = {55.1, 1.e-4, 1.e-10, 0.3, 0.1, 0.8};
//
//    canonicalizer.update(w);
//
//    CHECK_CANONICAL_FORM
//
//    expectedQ = {0, 5, 3, 4, 1, 2};
//    actualQ = canonicalizer.Q().indices();
//
//    CHECK(expectedQ.isApprox(actualQ));
}

TEST_CASE("Testing rationalize method.")
{
    const Index m = 10;
    const Index n = 60;

    MatrixXi Anum = random<int>(m, n);
    MatrixXi Aden = random<int>(m, n);
    MatrixXd A = Anum.cast<double>()/Aden.cast<double>();

    Canonicalizer canonicalizer(A);
    canonicalizer.rationalize(Aden.maxCoeff() * 10);

    CHECK_CANONICAL_FORM;
}
