// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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

#include "SaddlePointSolver.hpp"

// Optima includes
#include <Optima/Common/Exception.hpp>
#include <Optima/Common/Timing.hpp>
#include <Optima/Math/Canonicalizer.hpp>
#include <Optima/Math/Eigen/src/Cholesky/LDLT.h>

namespace Optima {

auto SaddlePointResult::operator+=(const SaddlePointResult& other) -> SaddlePointResult&
{
    success = success && other.success;
    time += other.time;
    return *this;
}

auto SaddlePointResult::operator+(SaddlePointResult other) -> SaddlePointResult
{
    other.success = success && other.success;
    other.time += time;
    return other;
}

struct SaddlePointSolver::Impl
{
    /// The Hessian matrix (used for optimization reasons)
    Vector H;

    /// The Jacobian matrix (used for optimization reasons)
    Matrix A;

    /// The right-hand side sub-vectors a and b (used for optimization reasons)
    Vector a, b;

    /// The canonicalizer of the Jacobian matrix `A`.
    Canonicalizer canonicalizer;

    /// The weights for the update of the canonical form.
    Vector w;

    /// The LDLT solver applied to `lhs_xb` to compute `xb`
    Eigen::LDLT<Matrix> ldlt;

    /// The number of rows and columns in A
    Index m, n;

    /// The number of basic, non-basic, and fixed variables.
    Index nb, nn, nf;

    /// Canonicalize the coefficient matrix \eq{A} of the saddle point problem.
    auto canonicalize(const SaddlePointMatrix& lhs) -> SaddlePointResult
    {
        // The time this method started
        Time begin = time();

        // Compute the canonical form of matrix A
        canonicalizer.compute(lhs.A);

        // Set the number of rows and columns in A
        m = lhs.A.rows();
        n = lhs.A.cols();

        // Allocate memory to H, A, a, b
        H.resize(n);
        A.resize(m, n);
        a.resize(n);
        b.resize(m);

        // The result of this method call
        SaddlePointResult res;
        res.time = elapsed(begin);

        return res;
    }

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(const SaddlePointMatrix& lhs, const Vector& weights) -> SaddlePointResult
    {
        // The time this method started
        Time begin = time();

        // Update the canonical form and the ordering of the variables
        canonicalizer.update(weights, lhs.ifixed);

        // Set the number of basic, non-basic, and fixed variables
        nb = canonicalizer.rows();
        nf = lhs.ifixed.size();
        nn = n - nb - nf;

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q();
        const auto& S = canonicalizer.S();

        // Assemble the matrices H and A with rows/columns ordering given by permutation matrix Q
        H.noalias() = rows(lhs.H, Q.indices());

        // Create auxiliary sub-matrix views
        auto Hb = H.head(nb);
        auto Hn = H.segment(nb, nn);
        auto Ab = A.topLeftCorner(nb, nb);
        auto An = A.middleCols(nb, nn).topRows(nb);
        auto Af = A.rightCols(nf);
        auto Sn = S.topLeftCorner(nb, nn);

        // Store the columns of A corresponding to fixed variables in Af
        Af = cols(lhs.A, lhs.ifixed);

        // Store the matrix product Sn*inv(Hn) in An
        An.noalias() = Sn * diag(inv(Hn));

        // Store the coefficient matrix to compute xb in Ab
        Ab.noalias() = Sn * tr(An);
        Ab.diagonal().noalias() += inv(Hb);

        // Compute the LDLT decomposition of `Ab`.
        ldlt.compute(Ab);

        // The result of this method call
        SaddlePointResult res;
        res.success = ldlt.info() == Eigen::Success;
        res.time = elapsed(begin);

        return res;
    }

    auto solve(const SaddlePointVector& rhs, SaddlePointVector& sol) -> SaddlePointResult
    {
        // The time this method started
        Time begin = time();

        // Alias to members of the saddle point vector solution.
        auto& x = sol.x;
        auto& y = sol.y;

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q();
        const auto& R = canonicalizer.R();
        const auto& S = canonicalizer.S();

        // Create auxiliary sub-matrix and sub-vector views
        auto Hb = H.head(nb);
        auto Hn = H.segment(nb, nn);
        auto An = A.middleCols(nb, nn).topRows(nb);
        auto Af = A.rightCols(nf);
        auto Sn = S.topLeftCorner(nb, nn);
        auto ab = a.head(nb);
        auto an = a.segment(nb, nn);
        auto af = a.tail(nf);

        // Reorder the sub-vectors a in the canonical order
        a.noalias() = rows(rhs.x, Q.indices());

        // Remove the contribution of fixed variables in b
        b.noalias() = rhs.y - Af*af;

        // Apply the regularize matrix to b
        b = R*b;

        // Resize the saddle point solution vector
        x.resize(n);
        y.resize(m);

        // Create views to the basic and non-basic blocks of vectors x
        auto xb = x.head(nb);
        auto xn = x.segment(nb, nn);
        auto xf = x.tail(nf);

        // Assemble the vector b to calculate xb in Ab*xb = b
        an.noalias() -= tr(Sn) * ab;
        an.noalias()  = an/Hn;
         b.noalias() -= Sn * an;

        // Compute the saddle point problem solution
        xb.noalias() = ldlt.solve(b);
         y.noalias() = ab - xb;
        xn.noalias() = an + tr(An) * xb;
        xb.noalias() = xb/Hb;
        xf.noalias() = af;
         y = tr(R)*y;

        // Permute back the variables x to their original ordering
        Q.applyThisOnTheLeft(x);

        // The result of this method call
        SaddlePointResult res;
        res.time = elapsed(begin);

        return res;
    }
};

SaddlePointSolver::SaddlePointSolver()
: pimpl(new Impl())
{}

SaddlePointSolver::SaddlePointSolver(const SaddlePointSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolver::~SaddlePointSolver()
{}

auto SaddlePointSolver::operator=(SaddlePointSolver other) -> SaddlePointSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolver::canonicalize(const SaddlePointMatrix& lhs) -> SaddlePointResult
{
    return pimpl->canonicalize(lhs);
}

auto SaddlePointSolver::decompose(const SaddlePointMatrix& lhs, const Vector& weights) -> SaddlePointResult
{
    return pimpl->decompose(lhs, weights);
}

auto SaddlePointSolver::solve(const SaddlePointVector& rhs, SaddlePointVector& sol) -> SaddlePointResult
{
    return pimpl->solve(rhs, sol);
}

} // namespace Optima
