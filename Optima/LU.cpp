// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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
#include <Eigen/src/LU/FullPivLU.h>

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

    /// The workspace for modified matrix U.
    Matrix U;

    /// The flags that indicate if an equation is linearly independent (non-zero value).
    Indices is_li;

    /// The rank of the lineary system, not of the coefficient matrix (depends on right-hand side vector!)
    Index rank;

    /// Construct a default Impl object.
    Impl()
    {}

    /// Construct an Impl object with given matrix.
    Impl(MatrixView A)
    {
        decompose(A);
    }

    /// Return true if empty.
    auto empty() const -> bool
    {
        return lu.matrixLU().size() == 0;
    }

    /// Compute the LU decomposition of the given matrix.
    auto decompose(MatrixView A) -> void
    {
        const auto m = A.rows();
        const auto n = A.cols();
        assert(n == m);
        lu.compute(A);
    }

    /// Solve the linear system `Ax = b` using the LU decomposition obtained with @ref decompose.
    auto solve(VectorView b, VectorRef x) -> void
    {
        x = b;
        solve(x);
    }

    /// Solve the linear system `Ax = b` using the LU decomposition obtained with @ref decompose.
    auto solve(VectorRef x) -> void
    {
        const auto& n  = lu.rows();
        const auto& P  = lu.permutationP();
        const auto& Q  = lu.permutationQ();
        const auto& M  = lu.matrixLU();
        const auto& Lv = M.triangularView<Eigen::UnitLower>();
        const auto& Uv = U.triangularView<Eigen::Upper>();

        assert(n == x.rows());

        P.applyThisOnTheLeft(x);
        x = Lv.solve(x);
        assembleU(x);
        x = Uv.solve(x);
        Q.applyThisOnTheLeft(x);
        Q.applyThisOnTheLeft(is_li);

        // TODO; In LU, x should have +inf or -inf to indicate extremely large steps and their directions. Then a line search would be used to find a reasonable step length/
    }

    /// Assemble the U matrix with given y, where y is the solution of L*y = P*b.
    auto assembleU(VectorRef y) -> void
    {
        const auto n = lu.rows();

        U.resize(n, n);
        U = lu.matrixLU();

        is_li.setOnes(n); // set all equations as linearly independent to start with

        // Skip the rest if there is only one equation.
        if(n == 1)
            return;

        const auto D = U.diagonal().cwiseAbs();
        const auto eps = std::numeric_limits<double>::epsilon();

        using std::max;
        using std::abs;

        rank = n; // start full rank, decrease as we go along through the diagonal of U (from the bottom!)
        for(auto i = 1; i <= n; ++i)
        {
            // Check diagonal entry is not very small compared to other entries
            // on same row of U and also right-hand side entry in y. The idea
            // is that if a diagonal entry is very small, but the other
            // coefficients along the same row of U is also very small and the
            // corresponding entry in y is equally small, then it is safe not
            // to discard this linear equation. Otherwise, we discard it, to
            // avoid extremely large values when we divide a larger number by
            // the diagonal pivot (very small).
            if(D[n - i] <= eps * abs(y[n - i]))
            {
                U.row(n - i).tail(n - i).fill(0.0); // avoid going to the lower triangular part
                U.col(n - i).head(n - i).fill(0.0);
                U(n - i, n - i) = 1.0; // diagonal entry is 1 while the rest entries along the same row/col are 0
                y[n - i] = 0.0; // the solution of the corresponding unknown should be zero
                is_li[n - i] = 0; // equation not linearly independent
                --rank;
            }
        }
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

auto LU::decompose(MatrixView A) -> void
{
    pimpl->decompose(A);
}

auto LU::solve(VectorView b, VectorRef x) -> void
{
    x = b;
    pimpl->solve(x);
}

auto LU::solve(VectorRef x) -> void
{
    pimpl->solve(x);
}

auto LU::rank() const -> Index
{
    return pimpl->rank;
}

auto LU::matrixLU() const -> MatrixView
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
