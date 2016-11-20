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

CanonicalMatrix::CanonicalMatrix()
{}

CanonicalMatrix::CanonicalMatrix(const Matrix& A)
{
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

	// Initialize the rank of matrix A
	m_rank = r;

	// Set the permutation matrices P and Q
	m_P = lu.permutationP();
	m_Q = lu.permutationQ();

	// Calculate the regularizer matrix R
	m_R = m_P;
	m_R.conservativeResize(r, m);
	m_R = Lbb.solve(m_R);
	m_R = Ubb.solve(m_R);

	// Calculate the inverse of the regularizer matrix R
	m_Rinv = m_P.transpose();
	m_Rinv.conservativeResize(m, r);
	m_Rinv = m_Rinv * Lbb;
	m_Rinv = m_Rinv * Ubb;

	// Calculate matrix S
	m_S = Ubn;
	m_S = Ubb.solve(m_S);
}

auto CanonicalMatrix::S() const -> const Matrix&
{
	return m_S;
}

auto CanonicalMatrix::R() const -> const Matrix&
{
	return m_R;
}

auto CanonicalMatrix::Rinv() const -> const Matrix&
{
	return m_Rinv;
}

auto CanonicalMatrix::P() const -> const PermutationMatrix&
{
	return m_P;
}

auto CanonicalMatrix::Q() const -> const PermutationMatrix&
{
	return m_Q;
}

auto CanonicalMatrix::rank() const -> Index
{
	return m_rank;
}

auto CanonicalMatrix::ili() const -> Indices
{
	PermutationMatrix Ptr = m_P.transpose();
	auto begin = Ptr.indices().data();
	return Indices(begin, begin + m_rank);
}

auto CanonicalMatrix::ibasic() const -> Indices
{
	auto begin = m_Q.indices().data();
	return Indices(begin, begin + m_rank);
}

auto CanonicalMatrix::inonbasic() const -> Indices
{
	auto begin = m_Q.indices().data();
	return Indices(begin + m_rank, begin + rows());
}

} // namespace Optima
