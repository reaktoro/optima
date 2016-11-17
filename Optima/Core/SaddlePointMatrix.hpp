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
struct SaddlePointMatrix;
struct SaddlePointMatrixCanonical;
struct SaddlePointVector;
struct SaddlePointVectorCanonical;

} // namespace Optima

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
        ColsAtCompileTime = Optima::Vector::ColsAtCompileTime,
        MaxRowsAtCompileTime = Optima::Vector::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = Optima::Vector::MaxColsAtCompileTime,
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
        ColsAtCompileTime = Optima::Vector::ColsAtCompileTime,
        MaxRowsAtCompileTime = Optima::Vector::MaxRowsAtCompileTime,
        MaxColsAtCompileTime = Optima::Vector::MaxColsAtCompileTime,
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
        const Index p = Z.rows();

        // Checking top rows [H -tr(A) -I]
        if(row < n)
        {
            // Block: H
            if(col < n) return row == col ? H[row] : 0.0;

            // Block: -tr(A)
            if(col < n + m) return -A(col - n, row);

            // Block: -I
            if(p > 0) return row == col - n - m ? -1.0 : 0.0;
        }

        // Checking middle rows [A 0 0]
        if(row < n + m)
        {
            // Block: A
            if(col < n) return A(row - n, col);

            // Block: 0
            return 0.0;
        }

        // Checking bottom rows [Z 0 X]
        if(p > 0)
        {
            row -= n + m;
            // Block: Z
            if(col < n) return col == row ? Z[col] : 0.0;

            // Block: 0
            if(col < n + m) return 0.0;

            // Block: X
            return row == col - n - m ? X[row] : 0.0;
        }

        return 0.0;
    }

    auto operator()(Index row, Index col) const -> Scalar { return coeff(row, col); }

    operator PlainObject() const
    {
        assert(valid());
        const Index n = H.rows();
        const Index m = A.rows();
        const Index p = Z.rows();
        const Index t = n + m + p;
        PlainObject res = zeros(t, t);
        res.topLeftCorner(n, n).diagonal() = H;
        res.middleRows(n, m).leftCols(n) = A;
        res.middleCols(n, m).topRows(n) = -tr(A);
        if(p > 0)
        {
            res.topRightCorner(n, n).diagonal() = -ones(n);
            res.bottomLeftCorner(n, n).diagonal() = Z;
            res.bottomRightCorner(n, n).diagonal() = X;
        }
        return res;
    }

    /// Return `true` if this SaddlePointMatrix instance is valid.
    auto valid() const -> bool
    {
        if(H.rows() != A.cols()) return false;
        if(Z.rows() && Z.rows() != H.rows()) return false;
        if(Z.rows() != X.rows()) return false;
        return true;
    }
};

/// A type used to describe a saddle point right-hand side vector.
struct SaddlePointVector : public Eigen::MatrixBase<SaddlePointVector>
{
    /// The saddle-point vector `x`.
    Vector x;

    /// The saddle-point vector `y`.
    Vector y;

    /// The saddle-point vector `z`.
    Vector z;

    EIGEN_DENSE_PUBLIC_INTERFACE(SaddlePointVector)

    auto rows() const -> Index { return x.rows() + y.rows() + z.rows(); }
    auto size() const -> Index { return rows(); }
    auto cols() const -> Index { return 1; }

    auto coeff(Index row) const -> Scalar
    {
        const Index n = x.rows();
        const Index m = y.rows();
        eigen_assert(row >= 0 && row < size());
        if(row < n) return x.operator[](row);
        if(row < n + m) return y[row - n];
        return z[row - n - m];
    }

    auto coeff(Index row, Index col) const -> Scalar
	{
    	eigen_assert(col == 0);
    	return coeff(row);
	}

    auto operator()(Index row) const -> Scalar { return coeff(row); }

    operator PlainObject() const
    {
        assert(valid());
        const Index n = x.rows();
        const Index m = y.rows();
        const Index p = z.rows();
        const Index t = n + m + p;
        PlainObject res(t);
        res.topRows(n) = x;
        res.middleRows(n, m) = y;
        res.bottomRows(p) = z;
        return res;
    }

    /// Return `true` if this SaddlePointVector instance is valid.
    auto valid() const -> bool
    {
        if(z.rows() && z.rows() != x.rows()) return false;
        return true;
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
        const Index m  = Bb.rows();
        const Index pb = Eb.rows();
        const Index ps = Es.rows();
        const Index pu = Eu.rows();
        const Index n  = nb + ns + nu;
        const Index p  = pb + ps + pu;
        return n + m + p;
    }

    auto cols() const -> Index { return rows(); }

    auto coeff(Index row, Index col) const -> Scalar
    {
        const Index nb = Gb.rows();
        const Index ns = Gs.rows();
        const Index nu = Gu.rows();
        const Index m  = Bb.rows();
        const Index pb = Eb.rows();
        const Index ps = Es.rows();
        const Index pu = Eu.rows();
        const Index n  = nb + ns + nu;
        const Index p  = pb + ps + pu;

        // Block: [G tr(B) E]
        if(row < n)
        {
            // Block: G
            if(col < n)
            {
                if(row < nb) return row == col ? Gb[row] : 0.0;
                if(row < nb + ns) return row == col ? Gs[row - nb] : 0.0;
                return row == col ? Gu[row - nb - ns] : 0.0;
            }

            // Block: tr(B)
            if(col < n + m)
            {
                col -= n;
                if(row < nb) return row == col ? Bb[row] : 0.0;
                if(row < nb + ns) return Bs(col, row - nb);
                return Bu(col, row - nb - ns);
            }

            // Block: E
            if(col < n + m + p)
            {
                col -= n + m;
                if(row < nb) return row == col ? Eb[row] : 0.0;
                if(row < nb + ns) return row == col ? Es[row - nb] : 0.0;
                return row == col ? Eu[row - nb - ns] : 0.0;
            }
        }
        // Block: [B 0 0]
        if(row < n + m)
        {
            // Block: B
            if(col < n)
            {
                row -= n;
                if(col < nb) return col == row ? Bb[row] : 0.0;
                if(col < nb + ns) return Bs(row, col - nb);
                if(col < n) return Bu(row, col - nb - ns);
            }

            // Block: 0
            return 0.0;
        }
        // Block: [E 0 E]
        if(row < n + m + p)
        {
            // Block E
            if(col < n)
            {
                row -= n + m;
                if(col != row) return 0.0;
                if(col < nb) return Eb[col];
                if(col < nb + ns) return Es[col - nb];
                return Eu[row - nb - ns];
            }
            // Block 0
            if(col < n + m)
                return 0.0;
            // Block: E
            if(col < n + m + p)
            {
                row -= n + m;
                col -= n + m;
                if(row != col) return 0.0;
                if(row < nb) return Eb[row];
                if(row < nb + ns) return Es[row - nb];
                return Eu[row - nb - ns];
            }
        }

        return 0.0;
    }

    auto operator()(Index row, Index col) const -> Scalar { return coeff(row, col); }

    operator PlainObject() const
    {
        assert(valid());

        const Index nb = Gb.rows();
        const Index ns = Gs.rows();
        const Index nu = Gu.rows();
        const Index m  = Bb.rows();
        const Index pb = Eb.rows();
        const Index ps = Es.rows();
        const Index pu = Eu.rows();
        const Index n  = nb + ns + nu;
        const Index p  = pb + ps + pu;
        const Index t  = n + m + p;

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

        if(pb) ETR.topRows(nb) = Eb;
        if(ps) ETR.middleRows(nb, ns) = Es;
        if(pu) ETR.bottomRows(nu) = Eu;

        if(pb) EBL.topRows(nb) = Eb;
        if(ps) EBL.middleRows(nb, ns) = Es;
        if(pu) EBL.bottomRows(nu) = Eu;

        if(pb) EBR.topRows(nb) = Eb;
        if(ps) EBR.middleRows(nb, ns) = Es;
        if(pu) EBR.bottomRows(nu) = Eu;

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
        if(Gb.rows() != Bb.rows()) return false;
        if(Bs.rows() && Bs.rows() != Bb.rows()) return false;
        if(Bu.rows() && Bu.rows() != Bb.rows()) return false;
        if(Bs.rows() && Bs.cols() != Gs.rows()) return false;
        if(Bu.rows() && Bu.cols() != Gu.rows()) return false;
        if(Eb.rows() && Eb.rows() != Gb.rows()) return false;
        if(Es.rows() && Es.rows() != Gs.rows()) return false;
        if(Eu.rows() && Eu.rows() != Gu.rows()) return false;
        return true;
    }
};

/// A type used to describe a canonical saddle point right-hand side vector.
struct SaddlePointVectorCanonical : public Eigen::MatrixBase<SaddlePointVectorCanonical>
{
    /// The canonical saddle point vector `x = [xb, xs, xu]`.
    Vector xb, xs, xu;

    /// The canonical saddle point vector `y`.
    Vector y;

    /// The canonical saddle point vector `z = [zb, zs, zu]`.
    Vector zb, zs, zu;


    EIGEN_DENSE_PUBLIC_INTERFACE(SaddlePointVectorCanonical)

    auto rows() const -> Index
    {
        const Index nb = xb.rows();
        const Index ns = xs.rows();
        const Index nu = xu.rows();
        const Index m  = y.rows();
        const Index pb = zb.rows();
        const Index ps = zs.rows();
        const Index pu = zu.rows();
        const Index n  = nb + ns + nu;
        const Index p  = pb + ps + pu;
        return n + m + p;
    }

    auto size() const -> Index { return rows(); }
    auto cols() const -> Index { return 1; }

    auto coeff(Index row) const -> Scalar
    {
        const Index nb = xb.rows();
        const Index ns = xs.rows();
        const Index nu = xu.rows();
        const Index m  = y.rows();
        const Index pb = zb.rows();
        const Index ps = zs.rows();
        const Index pu = zu.rows();
        const Index n  = nb + ns + nu;
        const Index p  = pb + ps + pu;

        // Block: a
        if(row < n)
        {
            if(row < nb) return xb[row];
            if(row < nb + ns) return xs[row - nb];
            return xu[row - nb - ns];
        }
        // Block: b
        if(row < n + m)
        {
            row -= n;
            return y[row];
        }
        // Block: c
        if(row < n + m + p)
        {
            row -= n + m;
            if(row < nb) return zb[row];
            if(row < nb + ns) return zs[row - nb];
            return zu[row - nb - ns];
        }

        return 0.0;
    }

    auto coeff(Index row, Index col) const -> Scalar
	{
    	eigen_assert(col == 0);
    	return coeff(row);
	}

    auto operator()(Index row) const -> Scalar { return coeff(row); }

    operator PlainObject() const
    {
        assert(valid());

        const Index nb = xb.rows();
        const Index ns = xs.rows();
        const Index nu = xu.rows();
        const Index m  = y.rows();
        const Index pb = zb.rows();
        const Index ps = zs.rows();
        const Index pu = zu.rows();
        const Index n  = nb + ns + nu;
        const Index p  = pb + ps + pu;
        const Index t  = n + m + p;

        PlainObject res(t);

        auto a = res.topRows(n);
        auto c = res.bottomRows(n);

        if(nb) a.topRows(nb) = xb;
        if(ns) a.middleRows(nb, ns) = xs;
        if(nu) a.bottomRows(nu) = xu;

        res.middleRows(n, m) = y;

        if(pb) c.topRows(nb) = zb;
        if(ps) c.middleRows(nb, ns) = zs;
        if(pu) c.bottomRows(nu) = zu;

        return res;
    }

    /// Return `true` if this SaddlePointMatrixCanonical instance is valid.
    auto valid() const -> bool
    {
        if(xb.rows() != zb.rows()) return false;
        if(xs.rows() != zs.rows()) return false;
        if(xu.rows() != zu.rows()) return false;
        return true;
    }
};

} // namespace Optima
