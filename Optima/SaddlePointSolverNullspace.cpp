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

#include "SaddlePointSolverNullspace.hpp"

// C++ includes
#include <cassert>

// Eigen includes
#include <Optima/deps/eigen3/Eigen/src/LU/PartialPivLU.h>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolverNullspace::Impl
{
    Vector aw;  ///< The workspace for the right-hand side vectors a and b
    Vector bw;  ///< The workspace for the right-hand side vectors a and b
    Matrix mat; ///< The matrix used as a workspace for the decompose and solve methods.
    Vector vec; ///< The vector used as a workspace for the decompose and solve methods

    Eigen::PartialPivLU<Matrix> lu; ///< The LU decomposition solver.

    /// Construct a default SaddlePointSolverNullspace::Impl instance.
    Impl(Index n, Index m)
    : mat(n + m, n + m), vec(n + m)
    {}

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    auto decompose(CanonicalSaddlePointMatrix args) -> void
    {
        // The dimension variables needed below
        const auto nbx = args.dims.nbx;
        const auto nnx = args.dims.nnx;

        // Views to the sub-matrices in Hxx = [Hbxbx Hbxnx; Hnxbx Hnxnx]
        auto Hbxbx = args.Hxx.topLeftCorner(nbx, nbx);
        auto Hbxnx = args.Hxx.topRightCorner(nbx, nnx);
        auto Hnxbx = args.Hxx.bottomLeftCorner(nnx, nbx);
        auto Hnxnx = args.Hxx.bottomRightCorner(nnx, nnx);

        // The matrix M where we setup the coefficient matrix of the equations
        auto M = mat.topLeftCorner(nnx, nnx);

        // Calculate the coefficient matrix M of the system of linear equations
        M.noalias() = Hnxnx;
        M += tr(args.Sbxnx) * Hbxbx * args.Sbxnx;
        M -= Hnxbx * args.Sbxnx;
        M -= tr(args.Sbxnx) * Hbxnx;

        // Compute the LU decomposition of M.
        if(nnx) lu.compute(M);
    }

    /// Solve the canonical saddle point problem.
    auto solve(CanonicalSaddlePointProblem args) -> void
    {
        // The dimension variables needed below
        const auto nbx = args.dims.nbx;
        const auto nnx = args.dims.nnx;
        const auto nbe = args.dims.nbe;
        const auto nbi = args.dims.nbi;
        const auto nne = args.dims.nne;
        const auto nni = args.dims.nni;

        // Auxiliary references
        auto Sbxnx = args.Sbxnx;
        auto Hxx   = args.Hxx;

        // Views to the sub-matrices in Hxx = [Hbxbx Hbxnx; Hnxbx Hnxnx]
        auto Hbxbx = Hxx.topLeftCorner(nbx, nbx);
        auto Hbxnx = Hxx.topRightCorner(nbx, nnx);
        auto Hnxbx = Hxx.bottomLeftCorner(nnx, nbx);

        // Update the vector aw (workspace for vector ax')
        aw = args.ax;

        // Views to the sub-vectors in ax = [abx, anx]
        auto abx = aw.head(nbx);
        auto anx = aw.tail(nnx);
        auto abe = abx.head(nbe);
        auto abi = abx.tail(nbi);
        auto ane = anx.head(nne);
        auto ani = anx.tail(nni);

        // Update the vector bw (workspace for vector b')
        bw = args.bbx;

        // Views to the sub-vectors in b'
        auto bbx = bw.head(nbx);
        auto bbe = bbx.head(nbe);
        auto bbi = bbx.tail(nbi);

        // Set ybx = abx (before abx is changed)
        args.ybx = abx;

        // Calculate abx' = abx - Hbxbx*bbx''
        abx -= Hbxbx*bbx;

        // Calculate anx' = anx - Hnxbx*bbx'' - tr(Sbxnx)*abx'
        anx -= Hnxbx*bbx + tr(Sbxnx)*abx;

        // Solve the system of linear equations
        if(nnx) anx.noalias() = lu.solve(anx);

        // Calculate xbx and store in abx
        abx.noalias() = bbx - Sbxnx*anx;

        // Finalize calculation of ybx
        args.ybx -= Hbxbx*abx + Hbxnx*anx;

        // Colect computed xx in aw
        args.xx = aw;
    }
};

SaddlePointSolverNullspace::SaddlePointSolverNullspace(Index n, Index m)
: pimpl(new Impl(n, m))
{}

SaddlePointSolverNullspace::SaddlePointSolverNullspace(const SaddlePointSolverNullspace& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolverNullspace::~SaddlePointSolverNullspace()
{}

auto SaddlePointSolverNullspace::operator=(SaddlePointSolverNullspace other) -> SaddlePointSolverNullspace&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolverNullspace::decompose(CanonicalSaddlePointMatrix args) -> void
{
    return pimpl->decompose(args);
}

auto SaddlePointSolverNullspace::solve(CanonicalSaddlePointProblem args) -> void
{
    return pimpl->solve(args);
}

} // namespace Optima
