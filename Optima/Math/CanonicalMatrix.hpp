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
class CanonicalMatrix : public Eigen::MatrixBase<CanonicalMatrix>
{
public:
    EIGEN_DENSE_PUBLIC_INTERFACE(CanonicalMatrix)

    /// Construct a default CanonicalMatrix instance.
    CanonicalMatrix();

    /// Construct a CanonicalMatrix instance with given matrix.
    CanonicalMatrix(const Matrix& A);

    /// Return the `S` matrix of the canonical representation.
    auto S() const -> const Matrix&;

    /// Return the `R` canonicalizer matrix of the canonicalization.
    auto R() const -> const Matrix&;

    /// Return the inverse of the `R` matrix of the canonicalization.
    auto Rinv() const -> const Matrix&;

    /// Return the `P` permutation matrix of the canonicalization.
    auto P() const -> const PermutationMatrix&;

    /// Return the `Q` permutation matrix of the canonicalization.
    auto Q() const -> const PermutationMatrix&;

    /// Return the rank of the original matrix.
    auto rank() const -> Index;

    /// Return the indices of the linearly independent rows of the original matrix.
    auto ili() const -> Indices;

    /// Return the indices of the basic components.
    auto ibasic() const -> Indices;

    /// Return the indices of the non-basic components.
    auto inonbasic() const -> Indices;

    /// Return the number of rows of the canonical matrix.
	auto rows() const -> Index { return m_rank; }

    /// Return the number of columns of the canonical matrix.
	auto cols() const -> Index { return m_Q.cols(); }

	/// Return an entry of the canonical matrix.
	auto coeff(Index row, Index col) const -> Scalar
	{
		const Index m = rows();
		const Index n = cols();
		eigen_assert(row < m && col < n);
		if(col < m) return row == col ? 1.0 : 0.0;
		return m_S(row, col - m);
	}

	/// Return an entry of the canonical matrix.
	auto operator()(Index row, Index col) const -> Scalar { return coeff(row, col); }

	/// Convert this CanonicalMatrix instance to a Matrix instance.
	operator PlainObject() const
	{
		const Index m = rows();
		const Index n = cols();
		PlainObject res(m, n);
		res << identity(m, m), m_S;
		return res;
	}

private:
    /// The matrix @f$ S @f$.
    Matrix m_S;

    /// The rank of the original matrix.
    Index m_rank;

    /// The permutation matrix @f$ P @f$.
    PermutationMatrix m_P;

    /// The permutation matrix @f$ Q @f$.
    PermutationMatrix m_Q;

    /// The canonicalizer matrix @f$ R @f$.
    Matrix m_R;

    /// The inverse of the canonicalizer matrix @f$ R @f$.
    Matrix m_Rinv;
};

} // namespace Optima
