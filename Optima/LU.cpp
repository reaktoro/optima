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

    /// The rank of matrix `A`
    Index rank;

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
        // The number of rows and cols of A
        const auto m = A.rows();
        const auto n = A.cols();

        assert(n == m);

        lu.compute(A);

        // Compute the rank of the matrix
        const auto eps = std::numeric_limits<double>::epsilon();
        const auto D = lu.matrixLU().diagonal().cwiseAbs();
        const auto maxdiagcoeff = D.maxCoeff();
        const auto threshold = maxdiagcoeff * eps * n;
        rank = n; // start full rank, decrease as we go along through the diagonal of U (from the bottom!)
        for(auto i = 1; i <= n; ++i)
            if(D[n - i] <= threshold)
                --rank; // current diagonal entry in U is either zero or residual (as a result of round-off errors)
            else break; // stop searching for zero or residual values along the diagonal of U
    }

    /// Solve the linear system `AX = B` using the calculated LU decomposition.
    auto solve(MatrixRef X) -> void
    {
        const auto n = lu.matrixLU().rows();

        assert(n == X.rows());

        const auto M = lu.matrixLU().topLeftCorner(rank, rank);
        const auto L = M.triangularView<Eigen::UnitLower>();
        const auto U = M.triangularView<Eigen::Upper>();
        const auto P = lu.permutationP();
        const auto Q = lu.permutationQ();

        auto Xt = X.topRows(rank);
        auto Xb = X.bottomRows(n - rank);

        P.applyThisOnTheLeft(X);
        Xt = L.solve(Xt);
        Xt = U.solve(Xt);
        Xb.fill(0.0);
        Q.applyThisOnTheLeft(X);
    }
};

LU::LU()
: pimpl(new Impl())
{}

LU::LU(MatrixConstRef A)
: pimpl(new Impl(A))
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

auto LU::rank() const -> Index
{
    return pimpl->rank;
}

auto LU::matrixLU() const -> MatrixConstRef
{
    return pimpl->lu.matrixLU();
}

auto LU::P() const -> PermutationMatrix
{
    return pimpl->lu.permutationP();
}

} // namespace Optima
