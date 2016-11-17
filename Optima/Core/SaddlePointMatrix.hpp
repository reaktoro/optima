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
		ColsAtCompileTime = Optima::Matrix::ColsAtCompileTime,
		MaxRowsAtCompileTime = Optima::Matrix::MaxRowsAtCompileTime,
		MaxColsAtCompileTime = Optima::Matrix::MaxColsAtCompileTime,
		CoeffReadCost = Optima::Matrix::CoeffReadCost
	};
};

template<>
struct traits<Optima::SaddlePointVector>
{
	typedef Eigen::Dense StorageKind;
	typedef Eigen::MatrixXpr XprKind;
	typedef Optima::Vector::Scalar Scalar;
	typedef Optima::Vector::Index Index;
	typedef Optima::Vector::PlainObject PlainObject;
	enum {
		Flags = Eigen::ColMajor,
		RowsAtCompileTime = Optima::Vector::RowsAtCompileTime,
		ColsAtCompileTime = 1,
		MaxRowsAtCompileTime = Optima::Vector::MaxRowsAtCompileTime,
		MaxColsAtCompileTime = 1,
		CoeffReadCost = Optima::Vector::CoeffReadCost
	};
};

template<>
struct traits<Optima::SaddlePointMatrixCanonical>
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

template<>
struct traits<Optima::SaddlePointVectorCanonical>
{
	typedef Eigen::Dense StorageKind;
	typedef Eigen::MatrixXpr XprKind;
	typedef Optima::Vector::Scalar Scalar;
	typedef Optima::Vector::Index Index;
	typedef Optima::Vector::PlainObject PlainObject;
	enum {
		Flags = Eigen::ColMajor,
		RowsAtCompileTime = Optima::Vector::RowsAtCompileTime,
		ColsAtCompileTime = 1,
		MaxRowsAtCompileTime = Optima::Vector::MaxRowsAtCompileTime,
		MaxColsAtCompileTime = 1,
		CoeffReadCost = Optima::Vector::CoeffReadCost
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
        const auto n = H.rows();
        const auto m = A.rows();
        const auto t = 2*n + m;
    	PlainObject res = zeros(t, t);
        res.topLeftCorner(n, n).diagonal() = H;
        res.topRightCorner(n, n).diagonal() = -ones(n);
        res.middleRows(n, m).leftCols(n) = A;
        res.middleCols(n, m).topRows(n) = -tr(A);
        res.bottomLeftCorner(n, n).diagonal() = Z;
        res.bottomRightCorner(n, n).diagonal() = X;
    	return res;
    }

    /// Return `true` if this SaddlePointMatrix instance is valid.
    auto valid() const -> bool
	{
    	return H.rows() == A.cols() &&
    		   X.rows() == Z.rows() &&
			  (X.rows() == H.rows() || X.rows() == 0);
	}
};

/// A type used to describe a saddle point right-hand side vector.
struct SaddlePointVector : public Eigen::MatrixBase<SaddlePointVector>
{
    /// The right-hand side vector `a`.
    Vector a;

    /// The right-hand side vector `b`.
    Vector b;

    /// The right-hand side vector `c`.
    Vector c;

    EIGEN_DENSE_PUBLIC_INTERFACE(SaddlePointVector)

    auto rows() const -> Index { return a.rows() + b.rows() + c.rows(); }
    auto size() const -> Index { return rows(); }
    auto cols() const -> Index { return 1; }

    auto coeff(Index row) const -> Scalar
	{
    	const Index n = a.rows();
    	const Index m = b.rows();
    	eigen_assert(row >= 0 && row < size());
    	if(row < n) return a.operator[](row);
    	if(row < n + m) return b[row - n];
    	return c[row - n - m];
	}

    auto operator()(Index row) const -> Scalar { return coeff(row); }

    operator PlainObject() const
    {
    	const Index n = a.rows();
    	const Index m = b.rows();
    	const Index t = 2*n + m;
    	PlainObject res(t);
    	res.topRows(n) = a;
    	res.middleRows(n, m) = b;
    	res.bottomRows(n) = c;
    	return res;
    }

    /// Return `true` if this SaddlePointVector instance is valid.
    auto valid() const -> bool
	{
    	return a.rows() == c.rows() || c.rows() == 0;
	}
};

/// A type used to describe a canonical saddle point coefficient matrix.
struct SaddlePointMatrixCanonical : public Eigen::MatrixBase<SaddlePointMatrixCanonical>
{
	/// The diagonal matrix `G = diag(Gb, Gs, Gu)` in the coefficient matrix.
    Vector Gb, Gs, Gu;

	/// The diagonal matrix `Bb` in the canonical coefficient matrix.
    Vector Bb;

	/// The matrix `B = [Bb Bs Bu]` in the canonical coefficient matrix.
    Matrix Bs, Bu;

    /// The diagonal matrix `E = diag(Eb, Es, Eu)` in the coefficient matrix.
    Vector Eb, Es, Eu;

    EIGEN_DENSE_PUBLIC_INTERFACE(SaddlePointMatrixCanonical)

	auto rows() const -> Index
	{
    	const Index nb = Gb.rows();
    	const Index ns = Gs.rows();
    	const Index nu = Gu.rows();
    	const Index n  = nb + ns + nu;
    	const Index m  = Bb.rows();
    	return 2*n + m;
	}

	auto cols() const -> Index { return rows(); }

	auto coeff(Index row, Index col) const -> Scalar
	{
    	const Index nb = Gb.rows();
    	const Index ns = Gs.rows();
    	const Index nu = Gu.rows();
    	const Index n  = nb + ns + nu;
    	const Index m  = Bb.rows();

		if(row < n && col < n) // Block: G
		{
			if(row < nb) return row == col ? Gb[row] : 0.0;
			if(row < nb + ns) return row == col ? Gs[row - nb] : 0.0;
			return row == col ? Gu[row - nb - ns] : 0.0;
		}
		if(row < n && col < n + m)  // Block: tr(B)
		{
			col -= n;
			if(row < nb) return row == col ? Bb[row] : 0.0;
			if(row < nb + ns) return Bs(col, row - nb);
			return Bu(col, row - nb - ns);
		}
		if(row < n)  // Block: E(top-right)
		{
			col -= n + m;
			if(row < nb) return row == col ? Eb[row] : 0.0;
			if(row < nb + ns) return row == col ? Es[row - nb] : 0.0;
			return row == col ? Eu[row - nb - ns] : 0.0;
		}
		if(row < n + m) // Block: B
		{
			row -= n;
			if(col < nb) return col == row ? Bb[row] : 0.0;
			if(col < nb + ns) return Bs(row, col - nb);
			if(col < n) return Bu(row, col - nb - ns);
			return 0.0;
		}
		if(col < n) // Block: E(bottom-left)
		{
			row -= n + m;
			if(col < nb) return row == col ? Eb[col] : 0.0;
			if(col < nb + ns) return row == col ? Es[col - nb] : 0.0;
			return row == col ? Eu[row - nb - ns] : 0.0;
		}
		if(row == col) // Block: E(bottom-right)
		{
			row -= n + m;
			if(row < nb) return Eb[row];
			if(row < nb + ns) return Es[row - nb];
			return Eu[row - nb - ns];
		}
		return 0.0;
	}

	auto operator()(Index row, Index col) const -> Scalar { return coeff(row, col); }

	operator PlainObject() const
	{
    	const Index nb = Gb.rows();
    	const Index ns = Gs.rows();
    	const Index nu = Gu.rows();
    	const Index n  = nb + ns + nu;
    	const Index m  = Bb.rows();
		const Index t  = 2*n + m;

		PlainObject res = zeros(t, t);

		auto G = res.topLeftCorner(n, n).diagonal();
		auto B = res.middleRows(n, m);
		auto BT = res.middleCols(n, m);
		auto ETR = res.topRightCorner(n, n).diagonal();
		auto EBL = res.bottomLeftCorner(n, n).diagonal();
		auto EBR = res.bottomRightCorner(n, n).diagonal();

		if(nb) G.topRows(nb) = Gb;
	    if(ns) G.middleRows(nb, ns) = Gs;
	    if(nu) G.bottomRows(nu) = Gu;

	    if(nb) ETR.topRows(nb) = Eb;
	    if(ns) ETR.middleRows(nb, ns) = Es;
	    if(nu) ETR.bottomRows(nu) = Eu;

	    if(nb) EBL.topRows(nb) = Eb;
	    if(ns) EBL.middleRows(nb, ns) = Es;
	    if(nu) EBL.bottomRows(nu) = Eu;

	    if(nb) EBR.topRows(nb) = Eb;
	    if(ns) EBR.middleRows(nb, ns) = Es;
	    if(nu) EBR.bottomRows(nu) = Eu;

	    if(nb) B.leftCols(nb) = diag(Bb);
	    if(ns) B.middleCols(nb, ns) = Bs;
	    if(nu) B.rightCols(nu) = Bu;

	    if(nb) BT.topRows(nb).diagonal() = Bb;
	    if(ns) BT.middleRows(nb, ns) = tr(Bs);
	    if(nu) BT.bottomRows(nu) = tr(Bu);

	    return res;
	}

    /// Return `true` if this SaddlePointMatrixCanonical instance is valid.
    auto valid() const -> bool
	{
    	return Gb.rows() == Bb.rows() &&
     		   Gs.rows() == Bs.cols() &&
    		   Gu.rows() == Bu.cols() &&
    		   Bs.rows() == Bb.rows() &&
			   Bu.rows() == Bb.rows() &&
			   Es.rows() == Eb.rows() &&
			   Eu.rows() == Eb.rows() &&
			  (Eb.rows() == Gb.rows() || Eb.rows() == 0) &&
			  (Es.rows() == Gs.rows() || Es.rows() == 0) &&
			  (Eu.rows() == Gu.rows() || Eu.rows() == 0);
	}
};

/// A type used to describe a canonical saddle point right-hand side vector.
struct SaddlePointVectorCanonical : public Eigen::MatrixBase<SaddlePointVectorCanonical>
{
    /// The right-hand side vector `a = [ab, as, au]` of the canonical problem.
    Vector ab, as, au;

    /// The right-hand side vector `b` of the canonical problem.
    Vector b;

    /// The right-hand side vector `c = [cb, cs, cu]` of the canonical problem.
    Vector cb, cs, cu;


    EIGEN_DENSE_PUBLIC_INTERFACE(SaddlePointVectorCanonical)

	auto rows() const -> Index
	{
    	const Index nb = ab.rows();
    	const Index ns = as.rows();
    	const Index nu = au.rows();
    	const Index n  = nb + ns + nu;
    	const Index m  = b.rows();
    	return 2*n + m;
	}

	auto size() const -> Index { return rows(); }
	auto cols() const -> Index { return 1; }

	auto coeff(Index row) const -> Scalar
	{
    	const Index nb = ab.rows();
    	const Index ns = as.rows();
    	const Index nu = au.rows();
    	const Index n  = nb + ns + nu;
    	const Index m  = b.rows();

		if(row < n) // Block: a
		{
			if(row < nb) return ab[row];
			if(row < nb + ns) return as[row - nb];
			return au[row - nb - ns];
		}
		if(row < n + m)  // Block: b
		{
			row -= n;
			return b[row];
		}
		else // Block: c
		{
			row -= n + m;
			if(row < nb) return cb[row];
			if(row < nb + ns) return cs[row - nb];
			return cu[row - nb - ns];
		}
	}

	auto operator()(Index row) const -> Scalar { return coeff(row); }

	operator PlainObject() const
	{
    	const Index nb = ab.rows();
    	const Index ns = as.rows();
    	const Index nu = au.rows();
    	const Index n  = nb + ns + nu;
    	const Index m  = b.rows();
		const Index t  = 2*n + m;

		PlainObject res(t);

		auto a = res.topRows(n);
		auto c = res.bottomRows(n);

		if(nb) a.topRows(nb) = ab;
	    if(ns) a.middleRows(nb, ns) = as;
	    if(nu) a.bottomRows(nu) = au;

		res.middleRows(n, m) = b;

		if(nb) c.topRows(nb) = cb;
	    if(ns) c.middleRows(nb, ns) = cs;
	    if(nu) c.bottomRows(nu) = cu;

	    return res;
	}

    /// Return `true` if this SaddlePointMatrixCanonical instance is valid.
    auto valid() const -> bool
	{
    	return (ab.rows() == cb.rows() || cb.rows() == 0) &&
     		   (as.rows() == cs.rows() || cs.rows() == 0) &&
    		   (au.rows() == cu.rows() || cu.rows() == 0);
	}
};

} // namespace Optima
