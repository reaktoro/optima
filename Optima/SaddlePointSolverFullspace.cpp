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

#include "SaddlePointSolverFullspace.hpp"

// C++ includes
#include <cassert>

// Eigen includes
#include <Optima/deps/eigen3/Eigen/src/LU/PartialPivLU.h>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverFullspace::Impl
{
    Matrix mat; ///< The matrix used as a workspace for the decompose and solve methods.
    Vector vec; ///< The vector used as a workspace for the decompose and solve methods

    Eigen::PartialPivLU<Matrix> lu; ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverFullspace::Impl instance.
    Impl(Index n, Index m)
    : mat(n + m, n + m), vec(n + m)
    {}

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        // The dimension variables needed below
        const auto nx  = args.dims.nx;
        const auto nbx = args.dims.nbx;
        const auto nnx = args.dims.nnx;

        // The identity matrix of dimension nbx
        auto Ibxbx = identity(nbx, nbx);

        // Create a view to the M block of the auxiliary matrix `mat` where the canonical saddle point matrix is defined
        auto M = mat.topLeftCorner(nx + nbx, nx + nbx);

        // Set the Ibb blocks in the canonical saddle point matrix
        M.bottomLeftCorner(nbx, nbx).noalias() = Ibxbx;
        M.topRightCorner(nbx, nbx).noalias() = Ibxbx;

        // Set the Sx and tr(Sx) blocks in the canonical saddle point matrix
        M.bottomRows(nbx).middleCols(nbx, nnx) = args.Sbxnx;
        M.rightCols(nbx).middleRows(nbx, nnx)  = tr(args.Sbxnx);

        // Set the G block of M on the bottom-right corner
        M.bottomRightCorner(nbx, nbx).setZero();

        // Set the H + D block of the canonical saddle point matrix
        M.topLeftCorner(nx, nx) = args.Hxx;

        // Compute the LU decomposition of M.
        lu.compute(M);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        // The dimension variables needed below
        const auto nx  = args.dims.nx;
        const auto nbx = args.dims.nbx;

        // View to the right-hand side vector r of the matrix equation
        auto r = vec.head(nx + nbx);

        // Assemble the right-hand side vector r = [ax bbx]
        r << args.ax, args.bbx;

        // Solve the system of linear equations using the LU decomposition of M.
        r.noalias() = lu.solve(r);

        // Get the result of xnx from r
        args.xx = r.head(nx);
        args.ybx = r.tail(nbx);
    }
};

SaddlePointSolverFullspace::SaddlePointSolverFullspace(Index n, Index m)
: pimpl(new Impl(n, m))
{}

SaddlePointSolverFullspace::SaddlePointSolverFullspace(const SaddlePointSolverFullspace& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolverFullspace::~SaddlePointSolverFullspace()
{}

auto SaddlePointSolverFullspace::operator=(SaddlePointSolverFullspace other) -> SaddlePointSolverFullspace&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolverFullspace::decompose(CanonicalSaddlePointMatrix args) -> void
{
    return pimpl->decompose(args);
}

auto SaddlePointSolverFullspace::solve(CanonicalSaddlePointProblem args) -> void
{
    return pimpl->solve(args);
}

} // namespace Optima
