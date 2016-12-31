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
#include <Optima/Math/Canonicalizer.hpp>
#include <Optima/Math/Eigen/src/LU/PartialPivLU.h>
using namespace Eigen;

namespace Optima {

SaddlePointResult::SaddlePointResult()
: m_success(true), m_time(0.0), m_start(Optima::time()), m_stop(m_start)
{}

auto SaddlePointResult::success(bool value) -> void
{
    m_success = value;
}

auto SaddlePointResult::success() const -> bool
{
    return m_success;
}

auto SaddlePointResult::time() const -> double
{
    return m_time;
}

auto SaddlePointResult::start() -> SaddlePointResult&
{
    m_time = 0.0;
    m_start = Optima::time();
    return *this;
}

auto SaddlePointResult::stop() -> SaddlePointResult&
{
    m_stop = Optima::time();
    m_time = elapsed(m_stop, m_start);
    return *this;
}

auto SaddlePointResult::operator+=(const SaddlePointResult& other) -> SaddlePointResult&
{
    m_success = m_success && other.m_success;
    m_time += other.m_time;
    return *this;
}

auto SaddlePointResult::operator+(SaddlePointResult other) -> SaddlePointResult
{
    other.m_success = m_success && other.m_success;
    other.m_time += m_time;
    return other;
}

struct SaddlePointSolver::Impl
{
    /// The canonicalizer of the Jacobian matrix `A`.
    Canonicalizer canonicalizer;

    /// Auxiliary matrices used in the decompose and solve methods.
    Matrix B, T, M;

    /// Auxiliary vectors used in the decompose and solve methods.
    Vector H, a, b, w;

    /// The LU solver aplied to the matrix Mb of dimension nb-by-nb to calculate Lb and Ub.
    PartialPivLU<Matrix> lu;

    /// The number of rows and columns in A
    Index m, n;

    /// The number of basic, non-basic, and fixed variables.
    Index nb, nn, nf;

    /// Canonicalize the coefficient matrix \eq{A} of the saddle point problem.
    auto canonicalize(const SaddlePointMatrix& lhs) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Set the number of rows and columns in A
        m = lhs.A.rows();
        n = lhs.A.cols();

        // Allocate auxiliary memory
        H.resize(n);
        B.resize(m, n);
        T.resize(n, m);
        M.resize(m, m);
        a.resize(n);
        b.resize(m);

        // Compute the canonical form of matrix A
        canonicalizer.compute(lhs.A);

        return res.stop();
    }

    /// Update the canonical form of the coefficient matrix \eq{A} of the saddle point problem.
    auto update(const SaddlePointMatrix& lhs, const Vector& weights) -> void
    {
        // Update the canonical form and the ordering of the variables
        canonicalizer.update(weights, lhs.fixed);

        // Update the number of fixed, basic, and non-basic variables
        nf = lhs.fixed.size();
        nb = canonicalizer.rows();
        nn = n - nb - nf;
    }

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(const SaddlePointMatrix& lhs, const Vector& weights) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Update the canonical form of the coefficient matrix A
        update(lhs, weights);

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q();
        const auto& S = canonicalizer.S();

        // Create auxiliary matrix views
        auto Hb = H.head(nb);
        auto Hn = H.segment(nb, nn);
        auto Bn = B.topLeftCorner(nb, nn);
        auto Tb = T.topLeftCorner(nn, nb);
        auto Mb = M.topLeftCorner(nb, nb);
        auto Sn = S.leftCols(nn);

        // Set `H` as the diagonal Hessian according to current canonical ordering
        H.noalias() = rows(lhs.H, Q.indices());

        // Compute the auxiliary matrices Bb and Bn
        Bn.noalias() = Sn * diag(inv(Hn));
        Tb.noalias() = tr(Sn) * diag(Hb);

        // Compute the matrix Mb
        Mb.noalias()  = Bn * Tb;
        Mb.noalias() += identity(nb, nb);

        // Compute the LU decomposition of `Mb`.
        lu.compute(Mb);

        return res.stop();
    }


    auto solve(const SaddlePointVector& rhs, SaddlePointVector& sol) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Alias to members of the saddle point vector solution.
        auto& x = sol.x;
        auto& y = sol.y;

        // Alias to the matrices of the canonicalization process
        const auto& Q = canonicalizer.Q();
        const auto& R = canonicalizer.R();
        const auto& S = canonicalizer.S();

        // Create auxiliary sub-matrix views
        auto Hb = H.head(nb);
        auto Hn = H.segment(nb, nn);
        auto Bn = B.topLeftCorner(nb, nn);
        auto Tb = T.topLeftCorner(nn, nb);
        auto Sn = S.leftCols(nn);
        auto Sf = S.rightCols(nf);

        // Initialize vectors a and b
        a = rhs.x;
        b = rhs.y;

        // Create views of the vector a = [ab an af] and b = [bb bz]
        auto ab = a.head(nb);
        auto an = a.segment(nb, nn);
        auto af = a.tail(nf);
        auto bb = b.head(nb);

        // Resize solution vectors x and y if needed
        x.resize(n);
        y.resize(m);

        // Create views of the vector x = [xb xn xf] and y = [yb yz]
        auto xb = x.head(nb);
        auto xn = x.segment(nb, nn);
        auto xf = x.tail(nf);
        auto yb = y.head(nb);

        // Reorder vector a in the canonical order
        Q.transpose().applyThisOnTheLeft(a);

//        Eigen::internal::set_is_malloc_allowed(false);
        // Apply the regularizer matrix to b
        bb = R*b;

        // Compute the saddle point problem solution
        an.noalias() -= tr(Sn)*ab;
        bb.noalias() -= Sf*af;
        bb.noalias() -= Bn*an;
        yb.noalias() = ab;
        xn.noalias() = an;
        xf.noalias() = af;
        xb.noalias() = lu.solve(bb);
        xn.noalias() += Tb*xb;
        xn.noalias()  = diag(inv(Hn))*xn;
        yb.noalias() -= diag(Hb)*xb;

//        Eigen::internal::set_is_malloc_allowed(true);

        // Compute the y vector without canonicalization
        y = tr(R)*yb;

        // Permute back the variables x to their original ordering
        Q.applyThisOnTheLeft(x);

        return res.stop();
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
