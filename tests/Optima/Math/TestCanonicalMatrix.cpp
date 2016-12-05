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

TEST_CASE("Testing CanonicalMatrix")
{
	const Index m = 4;
	const Index n = 10;

	Matrix A = random(m, n);

	CanonicalMatrix C(A);

	const Index r = C.rows();

	const auto& R = C.R();
	const auto& Rinv = C.Rinv();
	const auto& Q = C.Q();

	CHECK((R * Rinv).isApprox(identity(r, r)));
	CHECK((R * A * Q - C).norm() == approx(0.0));

	for(Index i = 0; i < r; ++i)
	{
		for(Index j = 0; j < n - r; ++j)
		{
			C.swap(i, j);
			CHECK((R * Rinv).isApprox(identity(r, r)));
			CHECK((R * A * Q - C).norm() == approx(0.0));
		}
	}
}

TEST_CASE("Testing CanonicalMatrix with two linearly dependent rows")
{
	const Index m = 4;
	const Index n = 10;

	Matrix A = random(m, n);
	A.row(2) = A.row(0) + 2*A.row(1);
	A.row(3) = A.row(1) - 2*A.row(2);

	CanonicalMatrix C(A);

	const Index r = C.rows();

	const auto& R = C.R();
	const auto& Rinv = C.Rinv();
	const auto& Q = C.Q();

	CHECK((R * Rinv).isApprox(identity(r, r)));
	CHECK((R * A * Q - C).norm() == approx(0.0));

	for(Index i = 0; i < r; ++i)
	{
		for(Index j = 0; j < n - r; ++j)
		{
			C.swap(i, j);
			CHECK((R * Rinv).isApprox(identity(r, r)));
			CHECK((R * A * Q - C).norm() == approx(0.0));
		}
	}
}

TEST_CASE("Testing the update method of the CanonicalMatrix class")
{
	const Matrix A = {
		{2,  1,  1,  1,  0,  0},
		{1,  0,  1,  3,  2,  3},
		{0,  0,  0,  1,  1,  1},
		{0,  1, -1, -1,  0, -2}
	};

	CanonicalMatrix C(A);

	const auto& R = C.R();
	const auto& Rinv = C.Rinv();
	const auto& Q = C.Q();

	const Index r = C.rows();

	CHECK(r == 3);
	CHECK((R * Rinv).isApprox(identity(r, r)));
	CHECK((R * A * Q - C).norm() == approx(0.0));

	Vector w = {55.1, 1.e-4, 1.e-10, 0.1, 0.5, 1e-2};

	C.update(w);

	CHECK((R * Rinv).isApprox(identity(r, r)));
	CHECK((R * A * Q - C).norm() == approx(0.0));

	Eigen::VectorXi expectedQ = {0, 4, 3, 5, 1, 2};

	CHECK(expectedQ.isApprox(Q.indices()));

	w = {55.1, 1.e-4, 1.e-10, 0.3, 0.1, 0.8};

	C.update(w);

	CHECK((R * Rinv).isApprox(identity(r, r)));
	CHECK((R * A * Q - C).norm() == approx(0.0));

	expectedQ = {0, 5, 3, 4, 1, 2};

	CHECK(expectedQ.isApprox(Q.indices()));

//	std::cout << "C = \n" << C << std::endl;
//	std::cout << "Q = \n" << Q.indices() << std::endl;
//
//	std::vector<std::string> species = {
//		"H2O", "H+", "OH-", "HCO3-", "CO2", "CO3--"
//	};
//
//	const auto& S = C.S();
//	std::cout << "C = \n" << C << std::endl;
//	std::cout << "R * A * Q = \n" << R * A * Q << std::endl;
//	std::cout << "Q = \n" << Q.indices() << std::endl;
//
//	for(Index i = 0; i < S.cols(); ++i)
//	{
//		std::cout << species[Q.indices()[i + S.rows()]] << " = ";
//		for(Index j = 0; j < S.rows(); ++j)
//			if(S(j, i))
//				std::cout  << " " << S(j, i) << "*" << species[Q.indices()[j]];
//		std::cout << std::endl;
//	}
//
//	for(Index i = 0; i < Q.cols(); ++i)
//		std::cout << species[Q.indices()[i]] << std::endl;
}
