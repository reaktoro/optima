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

#include "SaddlePointSolverDiagonalHessian.hpp"

// Optima includes
#include <Optima/Common/Exception.hpp>
#include <Optima/Math/Canonicalizer.hpp>
#include <Optima/Math/Eigen/src/Cholesky/LDLT.h>

namespace Optima {

struct SaddlePointSolverDiagonalHessian::Impl
{
    /// The coefficient matrix of the linear system used to compute `xb`
    Matrix lhs_xb;

    /// The right-hand side vector of the linear system used to compute `xb`
    Vector rhs_xb;

    /// The LDLT solver applied to `lhs_xb` to compute `xb`
    Eigen::LDLT<Matrix> ldlt;

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(const SaddlePointMatrixCanonical& clhs) -> void
    {
        // Alias to members of the canonical saddle point matrix
        const auto& G  = clhs.G;
        const auto& E  = clhs.E;
        const auto& Bb = clhs.Bb;
        const auto& Bn = clhs.Bn;
        const auto& nb = clhs.nb;
        const auto& nn = clhs.nn;
        const auto& ns = clhs.ns;
        const auto& nu = clhs.nu;

        // Create views to the basic, non-basic, stable, and unstable blocks of matrices G, E, and B.
        auto Gb =  G.head(nb);
        auto Gn =  G.tail(nn);
        auto Gs = Gn.head(ns);
        auto Gu = Gn.tail(nu);
        auto Eb =  E.head(nb);
        auto En =  E.tail(nn);
        auto Es = En.head(ns);
        auto Eu = En.tail(nu);
        auto Bs = Bn.leftCols(ns);
        auto Bu = Bn.rightCols(nu);

        // Define auxiliary light-weight matrix expressions
        auto GbEb = Gb - Eb;
        auto GsEs = Gs - Es;
        auto GuEu = Gu - Eu;
        auto BbBs = diag(inv(Bb)) * Bs;
        auto BbBu = diag(inv(Bb)) * Bu;

        // Assemble the left-hand side matrix of the linear system to compute `xb`
        lhs_xb = diag(inv(GbEb));
        lhs_xb += BbBs*diag(inv(GsEs))*tr(BbBs);
        lhs_xb += BbBu*diag(inv(GuEu))*tr(BbBu);

        // Compute the LDLT decomposition of `lhs_xb`.
        ldlt.compute(lhs_xb);
    }

    auto solve(const SaddlePointMatrixCanonical& clhs, const SaddlePointVector& crhs, SaddlePointVector& csol) -> void
    {
        // Alias to members of the canonical saddle point matrix
        const auto& G  = clhs.G;
        const auto& E  = clhs.E;
        const auto& Bb = clhs.Bb;
        const auto& Bn = clhs.Bn;
        const auto& nb = clhs.nb;
        const auto& nn = clhs.nn;
        const auto& ns = clhs.ns;
        const auto& nu = clhs.nu;

        // Alias to members of the canonical saddle point vector.
        const auto& r = crhs.x;
        const auto& s = crhs.y;
        const auto& t = crhs.z;

        // Create views to the basic, non-basic, stable, and unstable blocks of matrices G, E, X, and B.
        auto Gb =  G.head(nb);
        auto Gn =  G.tail(nn);
        auto Gs = Gn.head(ns);
        auto Gu = Gn.tail(nu);
        auto Eb =  E.head(nb);
        auto En =  E.tail(nn);
        auto Es = En.head(ns);
        auto Eu = En.tail(nu);
        auto Bs = Bn.leftCols(ns);
        auto Bu = Bn.rightCols(nu);

        // Create views to the basic, non-basic, stable, and unstable blocks of vectors r and t.
        auto rb =  r.head(nb);
        auto rn =  r.tail(nn);
        auto rs = rn.head(ns);
        auto ru = rn.tail(nu);
        auto tb =  t.head(nb);
        auto tn =  t.tail(nn);
        auto ts = tn.head(ns);
        auto tu = tn.tail(nu);

        // Alias to members of the canonical saddle point solution vector
        auto& x = csol.x;
        auto& y = csol.y;
        auto& z = csol.z;

        // The number of rows and columns of the canonical form of A
        const Index m = nb;
        const Index n = nb + nn;

        // Resize the saddle point solution vector
        x.resize(n);
        y.resize(m);
        z.resize(n);

        // Create views to the basic, non-basic, stable, and unstable blocks of vectors x, y, z.
        auto xb =  x.head(nb);
        auto xn =  x.tail(nn);
        auto xs = xn.head(ns);
        auto xu = xn.tail(nu);
        auto zb =  z.head(nb);
        auto zn =  z.tail(nn);
        auto zs = zn.head(ns);
        auto zu = zn.tail(nu);

        // Define auxiliary light-weight matrix expressions
        auto rbp  = rb - tb;
        auto rsp  = rs - ts;
        auto rup  = ru - Gu % (tu/Eu);
        auto tbp  = tb/Eb;
        auto tsp  = ts/Es;
        auto tup  = tu/Eu;
        auto sp   = s - Bu * tup;
        auto Bsp  = diag(inv(Bb)) * Bs;
        auto Bup  = diag(inv(Bb)) * Bu;
        auto spp  = sp/Bb;
        auto GbEb = Gb - Eb;
        auto GsEs = Gs - Es;
        auto GuEu = Gu - Eu;

        // Assemble the right-hand side vector of the linear system to compute `xb`
        rhs_xb = spp;
        rhs_xb.noalias() += Bsp*diag(inv(GsEs))*tr(Bsp)*rbp;
        rhs_xb.noalias() += Bup*diag(inv(GuEu))*tr(Bup)*rbp;
        rhs_xb.noalias() -= Bsp*(rsp/GsEs);
        rhs_xb.noalias() -= Bup*(rup/GuEu);

        // Compute the canonical variables x, y, z
        xb.noalias() = ldlt.solve(rhs_xb);
         y.noalias() = rbp - xb;
        xb.noalias() = xb/GbEb;
        xs.noalias() = (rsp - tr(Bsp)*y)/GsEs;
        zu.noalias() = (tr(Bup)*y - rup)/GuEu;
         y.noalias() = y/Bb;
        zb.noalias() = tbp - xb;
        zs.noalias() = tsp - xs;
        xu.noalias() = tup - zu;
    }
};

SaddlePointSolverDiagonalHessian::SaddlePointSolverDiagonalHessian()
: pimpl(new Impl())
{}

SaddlePointSolverDiagonalHessian::SaddlePointSolverDiagonalHessian(const SaddlePointSolverDiagonalHessian& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolverDiagonalHessian::~SaddlePointSolverDiagonalHessian()
{}

auto SaddlePointSolverDiagonalHessian::operator=(SaddlePointSolverDiagonalHessian other) -> SaddlePointSolverDiagonalHessian&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolverDiagonalHessian::decompose(const SaddlePointMatrixCanonical& clhs) -> void
{
    pimpl->decompose(clhs);
}

auto SaddlePointSolverDiagonalHessian::solve(const SaddlePointMatrixCanonical& clhs, const SaddlePointVector& crhs, SaddlePointVector& csol) -> void
{
    return pimpl->solve(clhs, crhs, csol);
}

} // namespace Optima
