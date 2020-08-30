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

// Optima includes
#include <Optima/IndexUtils.hpp>

// Eigen includes
#include <Optima/deps/eigen3/Eigen/src/LU/FullPivLU.h>

namespace Optima {

struct LU::Impl
{
    /// The base LU solver from Eigen library.
    Eigen::FullPivLU<Matrix> lusolver;

    /// The auxiliary matrix B in case X is provided as input/output when solving AX = B.
    Matrix B;

    /// The permutation matrix *P* of the LU decomposition *PAQ = LU*.
    Indices P;

    /// The permutation matrix *Q* of the LU decomposition *PAQ = LU*.
    Indices Q;

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
        return lusolver.size() == 0;
    }

    /// Compute the LU decomposition of the given matrix.
    auto decompose(MatrixConstRef A) -> void
    {
        // Check if number of rows and columns are equal
        assert(A.rows() == A.cols() && "Could not decompose the given matrix, "
            "which has different number of rows and columns.");

        // Perform a full pivoting decomposition of A, determining its rank
        lusolver.compute(A);

        // Store permutation matrices P and Q in indices form
        P = lusolver.permutationP().indices().cast<Index>();
        Q = lusolver.permutationQ().indices().cast<Index>();
    }

    /// Solve the linear system `AX = B` using the calculated LU decomposition when full rank.
    auto solveFullRank(MatrixConstRef B, MatrixRef X) -> void
    {
        X = lusolver.solve(B);
    }

    /// Solve the linear system `AX = B` using the calculated LU decomposition when rank deficient.
    auto solveRankDeficient(MatrixConstRef B, MatrixRef X) -> void
    {
        X = lusolver.solve(B);
        // const auto L = lusolver.matrixLU().triangularView<Eigen::UnitLower>();
        // const auto U = lusolver.matrixLU().triangularView<Eigen::Upper>();
        // const auto P = lusolver.permutationP();
        // const auto Q = lusolver.permutationQ();

        // P.applyThisOnTheLeft(X);
        // X = L.solve(X);
        // X = U.solve(X);



        // const auto L = lusolver.matrixLU().triangularView<Eigen::UnitLower>();
        // const auto P = lusolver.permutationP();
        // const auto Urr = Uw.topLeftCorner(rank, rank).triangularView<Eigen::Upper>();
        // const auto n = L.rows();

        // Xw.resize(X.rows(), X.cols());
        // auto Xr = Xw.topRows(rank);
        // auto Xs = Xw.bottomRows(n - rank);

        // P.applyThisOnTheLeft(X);
        // X = L.solve(X);

        // Xw = rows(X, jx);

        // Xr = Urr.solve(Xr);

        // // For the bottom part, corresponding to linearly dependent rows, set X to a quiet NaN.
        // Xs.fill(std::numeric_limits<double>::quiet_NaN());

        // rows(X, jx) = Xw;
    }

    /// Solve the linear system `AX = B` using the calculated LU decomposition.
    auto solve(MatrixConstRef B, MatrixRef X) -> void
    {
        solveRankDeficient(B, X);
        // const auto n = lusolver.matrixLU().rows();
        // if(rank == n) solveFullRank(X);
        // else solveRankDeficient(X);
    }

    /// Solve the linear system `AX = B` using the calculated LU decomposition.
    auto solve(MatrixRef X) -> void
    {
        solve(B = X, X);
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
    pimpl->solve(B, X);
}

auto LU::solve(MatrixRef X) -> void
{
    pimpl->solve(X);
}

auto LU::rank() const -> Index
{
    return pimpl->lusolver.rank();
}

auto LU::matrixLU() const -> MatrixConstRef
{
    return pimpl->lusolver.matrixLU();
}

auto LU::P() const -> IndicesConstRef
{
    return pimpl->P;
}

auto LU::Q() const -> IndicesConstRef
{
    return pimpl->Q;
}

} // namespace Optima
