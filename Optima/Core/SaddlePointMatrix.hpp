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

// C++ includes
#include <iostream>

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

// Forward declarations
struct SaddlePointMatrix;
struct SaddlePointMatrixCanonical;
struct SaddlePointVector;
struct SaddlePointVectorCanonical;

}

namespace Eigen {
namespace internal {

template<>
struct traits<Optima::SaddlePointMatrix>
{
	typedef Eigen::Dense StorageKind;
	typedef Eigen::MatrixXpr XprKind;
	typedef Optima::Matrix::Scalar Scalar;
	typedef Optima::Matrix::Index Index;
	typedef Optima::Matrix::PlainObject PlainObject;
	enum {
		Flags = Eigen::ColMajor,
		RowsAtCompileTime = Optima::Matrix::RowsAtCompileTime,
		ColsAtCompileTime = Optima::Matrix::RowsAtCompileTime,
		MaxRowsAtCompileTime = Optima::Matrix::MaxRowsAtCompileTime,
		MaxColsAtCompileTime = Optima::Matrix::MaxRowsAtCompileTime,
		CoeffReadCost = Optima::Matrix::CoeffReadCost
	};
};

} // namespace internal
} // namespace Eigen

namespace Optima {

/// A type used to describe a saddle point coefficient matrix.
struct SaddlePointMatrix : public Eigen::MatrixBase<SaddlePointMatrix>
{
	/// The diagonal matrix `H` in the coefficient matrix.
    Vector H;

	/// The matrix `A` in the coefficient matrix.
    Matrix A;

    /// The diagonal matrix `X` in the coefficient matrix.
    Vector X;

    /// The diagonal matrix `Z` in the coefficient matrix.
    Vector Z;

    EIGEN_DENSE_PUBLIC_INTERFACE(SaddlePointMatrix)

    auto rows() const -> Index { return H.rows() + A.rows() + X.rows(); }
    auto cols() const -> Index { return rows(); }

    auto coeff(Index row, Index col) const -> Scalar
	{
    	const Index n = H.rows();
    	const Index m = A.rows();

    	if(row < n && col < n)
    		return row == col ? H[row] : 0.0;
    	if(row < n && col < n + m)
    		return -A(col - n, row);
    	if(row < n)
    		return row == col - n - m ? -1.0 : 0.0;
    	if(row < n + m)
    		return col < n ? A(row - n, col) : 0.0;
    	if(col < n) return row - n - m == col ? Z[col] : 0.0;
    	if(col < n + m) return 0.0;
    	return row == col ? X[col - n - m] : 0.0;
	}

    auto operator()(Index row, Index col) const -> Scalar { return coeff(row, col); }

    operator PlainObject() const
    {
    	PlainObject res(rows(), cols());
    	for(Index i = 0; i < rows(); ++i)
    		for(Index j = 0; j < cols(); ++j)
    			res(i, j) = coeff(i, j);
    	return res;
    }
};

/// A type used to describe a saddle point right-hand side vector.
struct SaddlePointVector
{
    /// The right-hand side vector `a`.
    Vector a;

    /// The right-hand side vector `b`.
    Vector b;

    /// The right-hand side vector `c`.
    Vector c;
};

/// A type used to describe a canonical saddle point coefficient matrix.
struct SaddlePointMatrixCanonical
{
	/// The diagonal matrix `G = diag(Gb, Gs, Gu)` in the coefficient matrix.
    Vector Gb, Gs, Gu;

	/// The diagonal matrix `Bb` in the canonical coefficient matrix.
    Vector Bb;

	/// The matrix `B = [Bb Bs Bu]` in the canonical coefficient matrix.
    Matrix Bs, Bu;

    /// The diagonal matrix `E = diag(Eb, Es, Eu)` in the coefficient matrix.
    Vector Eb, Es, Eu;
};

/// A type used to describe a canonical saddle point right-hand side vector.
struct SaddlePointVectorCanonical
{
    /// The right-hand side vector `a = [ab, as, au]` of the canonical problem.
    Vector ab, as, au;

    /// The right-hand side vector `b` of the canonical problem.
    Vector b;

    /// The right-hand side vector `c = [cb, cs, cu]` of the canonical problem.
    Vector cb, cs, cu;
};

} // namespace Optima
