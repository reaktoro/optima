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
#include <Optima/Math/Eigen/src/Cholesky/LDLT.h>

namespace Optima {

struct SaddlePointSolver::Impl
{
    /// The coefficient matrix of the saddle point problem in canonical form.
    SaddlePointMatrixCanonical clhs;

    /// The right-hand side vector of the saddle point problem in canonical form.
    SaddlePointVectorCanonical crhs;

    /// The solution of the saddle point problem in canonical form.
    SaddlePointVectorCanonical csol;

    /// The canonicalizer of the coefficient matrix A.
    Canonicalizer canonicalizer;

    /// The indices of the basic variables.
    Indices ibasic;

    /// The indices of the non-basic variables.
    Indices inonbasic;

    /// The indices of the stable variables among the non-basic variables.
    Indices istable;

    /// The indices of the unstable variables among the non-basic variables.
    Indices iunstable;

    /// The auxiliary data to calculate the scaling of the saddle point problem.
    Vector G, E, X, Z, r, t;

    /// The coefficient matrix of the linear system used to compute `xb`
    Matrix lhsxb;

    /// The right-hand side vector of the linear system used to compute `xb`
    Vector rhsxb;

    /// The LDLT solver applied to `lhsxb` to compute `xb`
    Eigen::LDLT<Matrix> ldlt;

    /// Canonicalize the coefficient matrix \eq{A} of the saddle point problem.
    auto canonicalize(const SaddlePointMatrix& lhs) -> void
    {
        // Alias to members of the saddle point matrix
        const auto& A = lhs.A;

        // Compute the canonical form of matrix A
        canonicalizer.compute(A);

        // Reserve memory to index related members
        ibasic.reserve(A.rows());
        inonbasic.reserve(A.cols());
        istable.reserve(A.cols());
        iunstable.reserve(A.cols());
    }

    /// Decompose the coefficient matrix of the saddle point problem.
    auto decompose(const SaddlePointMatrix& lhs) -> void
    {
        // Alias to members of the saddle point matrix
        const auto& H = lhs.H;

        // Alias to members of the canonical saddle point matrix
        auto& Gb = clhs.Gb;
        auto& Gs = clhs.Gs;
        auto& Gu = clhs.Gu;
        auto& Bb = clhs.Bb;
        auto& Bs = clhs.Bs;
        auto& Bu = clhs.Bu;
        auto& Eb = clhs.Eb;
        auto& Es = clhs.Es;
        auto& Eu = clhs.Eu;

        // Update vectors X and Z
        X.noalias() = lhs.X;
        Z.noalias() = lhs.Z;

        // Compute the scaled matrices G and E
        G.noalias() =  X % H % X;
        E.noalias() = -X % Z;

        // Update the canonical form by selecting basic variables with largest X values
        canonicalizer.update(X);

        // Extract the S matrix, where `R*A*Q = [I S]`
        const auto& S = canonicalizer.S();
        const auto& C = canonicalizer.C();

        // Update the indices of basic and non-basic variables
        ibasic = canonicalizer.ibasic();
        inonbasic = canonicalizer.inonbasic();

        // Update the indices of stable and unstable non-basic variables
        istable.clear();
        iunstable.clear();
        for(Index i = 0; i < inonbasic.size(); ++i)
        {
            const double j = inonbasic[i];
            const double Xj = X[j];
            const double Zj = Z[j];
            if(std::abs(Xj) > std::abs(Zj))
                istable.push_back(i);
            else iunstable.push_back(i);
        }

        // Create a view to the basic and non-basic entries of X
        auto Xb = rows(X, ibasic);
        auto Xn = rows(X, inonbasic);

        // Assemble the matrix G in the canonical saddle point matrix
        Gb.noalias() = rows(G, ibasic);
        Gs.noalias() = rows(rows(G, inonbasic), istable);
        Gu.noalias() = rows(rows(G, inonbasic), iunstable);

        // Assemble the matrix B in the canonical saddle point matrix
        Bb.noalias() = Xb;
        Bs.conservativeResize(ibasic.size(), istable.size());
        Bu.conservativeResize(ibasic.size(), iunstable.size());
        for(Index i = 0; i < istable.size(); ++i)
            Bs.col(i).noalias() = S.col(istable[i]) * Xn[istable[i]];
        for(Index i = 0; i < iunstable.size(); ++i)
            Bu.col(i).noalias() = S.col(iunstable[i]) * Xn[iunstable[i]];

        // Assemble the matrix E in the canonical saddle point matrix
        Eb.noalias() = rows(E, ibasic);
        Es.noalias() = rows(rows(E, inonbasic), istable);
        Eu.noalias() = rows(rows(E, inonbasic), iunstable);

        // Define auxiliary light-weight matrix expressions
        auto GbEb = Gb - Eb;
        auto GsEs = Gs - Es;
        auto GuEu = Gu - Eu;
        auto BbBs = diag(inv(Bb)) * Bs;
        auto BbBu = diag(inv(Bb)) * Bu;

        // Assemble the left-hand side matrix of the linear system to compute `xb`
        lhsxb = diag(inv(GbEb));
        lhsxb += BbBs*diag(inv(GsEs))*tr(BbBs);
        lhsxb += BbBu*diag(inv(GuEu))*tr(BbBu);

        // Compute the LDLT decomposition of `lhsxb`.
        ldlt.compute(lhsxb);
    }

    auto solve(const SaddlePointVector& rhs) -> SaddlePointVector
    {
        // Alias to members of the saddle point vector
        const auto& a = rhs.x;
        const auto& b = rhs.y;
        const auto& c = rhs.z;

        // Alias to members of the canonical saddle point matrix
        auto& Gb = clhs.Gb;
        auto& Gs = clhs.Gs;
        auto& Gu = clhs.Gu;
        auto& Bb = clhs.Bb;
        auto& Bs = clhs.Bs;
        auto& Bu = clhs.Bu;
        auto& Eb = clhs.Eb;
        auto& Es = clhs.Es;
        auto& Eu = clhs.Eu;

        // Alias to members of the canonical saddle point vector
        auto& rb = crhs.xb;
        auto& rs = crhs.xs;
        auto& ru = crhs.xu;
        auto&  s = crhs.y ;
        auto& tb = crhs.zb;
        auto& ts = crhs.zs;
        auto& tu = crhs.zu;

        // Alias to members of the canonical solution vector
        auto& cxb = csol.xb;
        auto& cxs = csol.xs;
        auto& cxu = csol.xu;
        auto& cy  = csol.y;
        auto& czb = csol.zb;
        auto& czs = csol.zs;
        auto& czu = csol.zu;

        // Get the R matrix, where `R*A*Q = [I S]`
        const auto& R = canonicalizer.R();

        // The number of rows and columns of the canonical matrix A
        const Index m = canonicalizer.rows();
        const Index n = canonicalizer.cols();

        // Update scaled right-hand side vectors r and t
        r.noalias() =  X % a;
        t.noalias() = -c;

        // Assemble the canonical right-hand side saddle point vector
        rb.noalias() = rows(r, ibasic);
        rs.noalias() = rows(rows(r, inonbasic), istable);
        ru.noalias() = rows(rows(r, inonbasic), iunstable);
         s.noalias() = R * b;
        tb.noalias() = rows(t, ibasic);
        ts.noalias() = rows(rows(t, inonbasic), istable);
        tu.noalias() = rows(rows(t, inonbasic), iunstable);

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
        rhsxb = spp;
        rhsxb.noalias() += Bsp*diag(inv(GsEs))*tr(Bsp)*rbp;
        rhsxb.noalias() += Bup*diag(inv(GuEu))*tr(Bup)*rbp;
        rhsxb.noalias() -= Bsp*(rsp/GsEs);
        rhsxb.noalias() -= Bup*(rup/GuEu);

        // Compute the canonical variables x, y, z
        cxb.noalias() = ldlt.solve(rhsxb);
         cy.noalias() = rbp - cxb;
        cxb.noalias() = cxb/GbEb;
        cxs.noalias() = (rsp - tr(Bsp)*cy)/GsEs;
        czu.noalias() = (tr(Bup)*cy - rup)/GuEu;
         cy.noalias() = cy/Bb;
        czb.noalias() = tbp - cxb;
        czs.noalias() = tsp - cxs;
        cxu.noalias() = tup - czu;

        SaddlePointVector solution;
        solution.x.resize(n);
        solution.y.resize(m);
        solution.z.resize(n);

        auto xb = rows(solution.x, ibasic);
        auto xn = rows(solution.x, inonbasic);
        auto xs = rows(xn, istable);
        auto xu = rows(xn, iunstable);
        auto& y = solution.y;
        auto zb = rows(solution.z, ibasic);
        auto zn = rows(solution.z, inonbasic);
        auto zs = rows(zn, istable);
        auto zu = rows(zn, iunstable);

        auto Xb = rows(X, ibasic);
        auto Xn = rows(X, inonbasic);
        auto Xs = rows(Xn, istable);
        auto Xu = rows(Xn, iunstable);

        auto Zb = rows(Z, ibasic);
        auto Zn = rows(Z, inonbasic);
        auto Zs = rows(Zn, istable);
        auto Zu = rows(Zn, iunstable);

        xb.noalias() =  Xb % cxb;
        xs.noalias() =  Xs % cxs;
        xu.noalias() =  Xu % cxu;
         y.noalias() = -tr(R) * cy;
        zb.noalias() =  Zb % czb;
        zs.noalias() =  Zs % czs;
        zu.noalias() =  Zu % czu;

        return solution;
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

auto SaddlePointSolver::canonicalize(const SaddlePointMatrix& lhs) -> void
{
    pimpl->canonicalize(lhs);
}

auto SaddlePointSolver::decompose(const SaddlePointMatrix& lhs) -> void
{
    pimpl->decompose(lhs);
}

auto SaddlePointSolver::solve(const SaddlePointVector& rhs) -> SaddlePointVector
{
    return pimpl->solve(rhs);
}

} // namespace Optima
