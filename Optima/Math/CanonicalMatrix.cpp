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

#include "CanonicalMatrix.hpp"

// Eigen includes
#include <Optima/Common/Exception.hpp>
#include <Optima/Math/Eigen/LU>

namespace Optima {

auto canonicalize(const Matrix& A) -> CanonicalMatrix
{
	// The canonical form of A
	CanonicalMatrix res;

	// The number of rows and columns of A
	const Index m = A.rows();
	const Index n = A.cols();

	// Check if number of columns is greater/equal than number of rows
	Assert(n >= m, "Could not canonicalize the given matrix.",
		"The given matrix has more rows than columns.");

	// Compute the full-pivoting LU of A so that P*A*Q = L*U
	Eigen::FullPivLU<Matrix> lu(A);

	// Get the rank of matrix A
	const Index r = lu.rank();

	// Get the LU factors of matrix A
	const auto Lbb = lu.matrixLU().topLeftCorner(r, r).triangularView<Eigen::UnitLower>();
	const auto Ubb = lu.matrixLU().topLeftCorner(r, r).triangularView<Eigen::Upper>();
	const auto Ubn = lu.matrixLU().topRightCorner(r, n - r);

	// Set the rank of matrix A
	res.rank = r;

	// Set the permutation matrices P and Q
	res.P = lu.permutationP();
	res.Q = lu.permutationQ();

	// Calculate the regularizer matrix R
	res.R = res.P;
	res.R.conservativeResize(r, m);
	res.R = Lbb.solve(res.R);
	res.R = Ubb.solve(res.R);

	// Calculate the inverse of the regularizer matrix R
	res.invR = res.P.transpose();
	res.invR.conservativeResize(m, r);
	res.invR = res.invR * Lbb;
	res.invR = res.invR * Ubb;

	// Calculate matrix S
	res.S = Ubn;
	res.S = Ubb.solve(res.S);

	return res;
}

} // namespace Optima
