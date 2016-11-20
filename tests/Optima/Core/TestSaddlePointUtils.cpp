// Optima is a C++ library for numerical sol of linear and nonlinear programing problems.
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
#include <Optima/Math/Eigen/LU>

TEST_CASE("Testing SaddlePointUtils")
{
	const Index m = 4;
	const Index n = 10;

	Matrix A = random(m, n);

	Eigen::FullPivLU<Matrix> lu(A);

	auto r = lu.rank();
	const PermutationMatrix P = lu.permutationP();
	const PermutationMatrix Q = lu.permutationQ();
	const Matrix L = lu.matrixLU().topLeftCorner(r, r).triangularView<Eigen::UnitLower>();
	const Matrix U = lu.matrixLU().topRightCorner(r, n).triangularView<Eigen::Upper>();
//	auto U = lu.matrixLU().topLeftCorner(r, r).triangularView<Eigen::Upper>();
//	auto V = lu.matrixLU().topRightCorner(r, n - r);


	std::cout << "P*A*Q = \n" << P*A*Q << std::endl;
	std::cout << "L*U = \n" << L*U << std::endl;

	CanonicalMatrix C = canonicalize(A);

	std::cout << "S = \n" << C.S << std::endl;
	std::cout << "R = \n" << C.R << std::endl;
	std::cout << "R*inv(R) = \n" << C.R * C.invR << std::endl;

	std::cout << "C = \n" << Matrix(C) << std::endl;
	std::cout << "R*A*Q = \n" << C.R * A * C.Q << std::endl;
	std::cout << "C = \n" << C - Matrix(C) << std::endl;

}
