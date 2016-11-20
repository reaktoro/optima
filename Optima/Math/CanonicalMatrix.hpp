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

#pragma once

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

// Forward declarations
struct CanonicalMatrix;

} // namespace Optima

namespace Eigen {
namespace internal {

template<>
struct traits<Optima::CanonicalMatrix>
{
    typedef Eigen::Dense StorageKind;
    typedef Eigen::MatrixXpr XprKind;
    typedef Optima::Matrix::Scalar Scalar;
    typedef Optima::Matrix::Index Index;
    typedef Optima::Matrix::PlainObject PlainObject;
    enum {
        Flags = Eigen::ColMajor,
        RowsAtCompileTime = Optima::Matrix::RowsAtCompileTime,
        ColsAtCompileTime = Optima::Matrix::ColsAtCompileTime,
        MaxRowsAtCompileTime = Optima::Matrix::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = Optima::Matrix::MaxColsAtCompileTime,
        CoeffReadCost = Optima::Matrix::CoeffReadCost
    };
};

} // namespace internal
} // namespace Eigen

namespace Optima {

/// A type used to describe a matrix @f$ A @f$ in canonical form.
/// The canonical form of a matrix @f$ A @f$ is represented as:
/// @f[
/// C = RAQ = \begin{bmatrix}I & S\end{bmatrix},
/// @f]
/// where @f$ Q @f$ is a permutation matrix, and @f$ R @f$ is the
/// *canonicalizer matrix* of @f$ A @f$.
struct CanonicalMatrix : public Eigen::MatrixBase<CanonicalMatrix>
{
    /// The matrix @f$ S @f$.
    Matrix S;

    /// The rank of the original matrix.
    Index rank;

    /// The permutation matrix @f$ P @f$.
    PermutationMatrix P;

    /// The permutation matrix @f$ Q @f$.
    PermutationMatrix Q;

    /// The canonicalizer matrix @f$ R @f$.
    Matrix R;

    /// The inverse of the canonicalizer matrix @f$ R @f$.
    Matrix invR;

    EIGEN_DENSE_PUBLIC_INTERFACE(CanonicalMatrix)

	auto rows() const -> Index { return rank; }
	auto cols() const -> Index { return Q.cols(); }

	auto coeff(Index row, Index col) const -> Scalar
	{
		const Index m = rows();
		const Index n = cols();
		eigen_assert(row < m && col < n);
		if(col < m) return row == col ? 1.0 : 0.0;
		return S(row, col - m);
	}

	auto operator()(Index row, Index col) const -> Scalar { return coeff(row, col); }

	operator PlainObject() const
	{
		const Index m = rows();
		const Index n = cols();
		PlainObject res(m, n);
		res << identity(m, m), S;
		return res;
	}
};

auto canonicalize(const Matrix& A) -> CanonicalMatrix;

} // namespace Optima
