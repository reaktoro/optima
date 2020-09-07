// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
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

#include "LU.hpp"

// C++ includes
#include <cassert>

// Eigen includes
#include <Optima/deps/eigen3/Eigen/src/LU/FullPivLU.h>

// Optima includes
#include <Optima/Macros.hpp>

namespace Optima {

struct LU::Impl
{
    //======================================================================
    // Note: The full pivoting strategy is needed at the moment to resolve
    // singular matrices. Using a partial pivoting scheme via PartialPivLU
    // would need to be combined with a search for linearly dependent rows in
    // the produced upper triangular matrix U.
    //======================================================================

    /// The base LU solver from Eigen library.
    Eigen::FullPivLU<Matrix> lu;

    /// The workspace for matrix Ab = inv(B)*A where B = diag(inv(b*)), b*[i] = b[i] if b[i] != 0 else 1.
    Matrix Ab;

    /// Construct a default Impl object.
    Impl()
    {}

    /// Construct an Impl object with given matrix.
    Impl(MatrixConstRef A)
    {
        decompose(A);
    }

    /// Return true if empty.
    auto empty() const -> bool
    {
        return lu.matrixLU().size() == 0;
    }

    /// Compute the LU decomposition of the given matrix.
    auto decompose(MatrixConstRef A) -> void
    {
        const auto m = A.rows();
        const auto n = A.cols();
        assert(n == m);
        lu.compute(A);
    }

    /// Solve the linear system `AX = B` using the LU decomposition obtained with @ref decompose.
    auto solve(MatrixConstRef B, MatrixRef X) -> void
    {
        X = B;
        solve(X);
    }

    /// Solve the linear system `AX = B` using the LU decomposition obtained with @ref decompose.
    auto solve(MatrixRef X) -> void
    {
        const auto n = lu.rows();
        const auto r = rank();
        assert(n == X.rows());
        assert(r <= n);
        const auto P = lu.permutationP();
        const auto Q = lu.permutationQ();
        const auto M = lu.matrixLU().topLeftCorner(r, r);
        const auto L = M.triangularView<Eigen::UnitLower>();
        const auto U = M.triangularView<Eigen::Upper>();
        auto Xt = X.topRows(r);
        auto Xb = X.bottomRows(n - r);

        P.applyThisOnTheLeft(X);
        Xt = L.solve(Xt);
        Xt = U.solve(Xt);
        Xb.fill(0.0);
        Q.applyThisOnTheLeft(X);
    }

    /// Return the rank of the last LU decomposed matrix.
    auto rank() const -> Index
    {
        const auto LU = lu.matrixLU();
        const auto n = LU.rows();
        const auto D = LU.diagonal().cwiseAbs();
        const auto eps = std::numeric_limits<double>::epsilon();

        Index r = n; // start full rank, decrease as we go along through the diagonal of U (from the bottom!)
        for(auto i = 1; i <= n; ++i)
            if(D[n - i] <= eps * norminf(LU.col(n - i).head(n - i)))
                --r; // current diagonal entry in U is very small compared to others on the same column.
            else break; // stop, because from now on there are only large enough diagonal values.

        return r;
    }

    /// Solve the linear system `Ax = b` with a scaling strategy for increased robustness.
    auto solveWithScaling(MatrixConstRef A, VectorConstRef b, VectorRef x) -> void
    {
        const auto m = A.rows();
        const auto n = A.cols();

        assert(n == m);
        assert(n == b.rows());
        assert(n == x.rows());

        //======================================================================
        // IMPORTANT:
        // Any small value in `b` is assumed here to be meaningful. This means that
        // small values as a result of residual round off-error should have been
        // cleaned off before this method is executed. Otherwise, this small
        // value may produce incorrect solutions and wrong behavior when
        // identifying linearly dependent equations.
        //======================================================================
        auto invB = x; // use x as workspace for invB

        invB = (b.array() == 0.0).select(1.0, b.cwiseInverse());

        Ab.noalias() = invB.asDiagonal() * A;

        decompose(Ab);

        auto eb = x; // use x as workspace for eb where eb[i] = 1 if b[i] != 0 else 0

        eb = (b.array() == 0.0).select(0.0, ones(n));

        solve(x); // equivalent to solve(eb, x)
    }
};

LU::LU()
: pimpl(new Impl())
{}

LU::LU(const LU& other)
: pimpl(new Impl(*other.pimpl))
{}

LU::~LU()
{}

auto LU::operator=(LU other) -> LU&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto LU::empty() const -> bool
{
    return pimpl->empty();
}

auto LU::decompose(MatrixConstRef A) -> void
{
    pimpl->decompose(A);
}

auto LU::solve(MatrixConstRef B, MatrixRef X) -> void
{
    X = B;
    pimpl->solve(X);
}

auto LU::solve(MatrixRef X) -> void
{
    pimpl->solve(X);
}

auto LU::solveWithScaling(MatrixConstRef A, VectorConstRef b, VectorRef x) -> void
{
    pimpl->solveWithScaling(A, b, x);
}

auto LU::rank() const -> Index
{
    return pimpl->rank();
}

auto LU::matrixLU() const -> MatrixConstRef
{
    return pimpl->lu.matrixLU();
}

auto LU::P() const -> PermutationMatrix
{
    return pimpl->lu.permutationP();
}

auto LU::Q() const -> PermutationMatrix
{
    return pimpl->lu.permutationP();
}

} // namespace Optima
