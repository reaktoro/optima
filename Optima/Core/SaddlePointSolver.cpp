// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2016 Allan Leal
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
#include <Optima/Math/CanonicalMatrix.hpp>
#include <Optima/Math/Eigen/LU>

namespace Optima {

struct SaddlePointSolverDiagonal
{
    /// The flag that indicates if matrix A is constant across several `solve` calls.
    bool constA = false;

    /// Auxiliary data for the LU decomposition of the saddle point matrix.
    Vector Lb;
    Matrix Ts;
    Matrix Tu;
    Matrix Ls;
    Matrix Tb;
    Matrix Lu;
    Matrix Ub;
    Vector ub;
    Vector us;
    Vector uu;
    Vector vb;
    Vector r;
    Vector invGsEs;
    Vector invEuGu;
    Vector invEucu;

    auto solve(const SaddlePointProblemCanonical& problem, SaddlePointVectorCanonical& solution) -> void
    {
        // Auxiliary alias to problem data members
        const auto& Gb = problem.lhs.Gb;
        const auto& Gs = problem.lhs.Gs;
        const auto& Gu = problem.lhs.Gu;
        const auto& Bb = problem.lhs.Bb;
        const auto& Bs = problem.lhs.Bs;
        const auto& Bu = problem.lhs.Bu;
        const auto& Eb = problem.lhs.Eb;
        const auto& Es = problem.lhs.Es;
        const auto& Eu = problem.lhs.Eu;
        const auto& ab = problem.rhs.db;
        const auto& as = problem.rhs.ds;
        const auto& au = problem.rhs.du;
        const auto& b  = problem.rhs.e;
        const auto& cb = problem.rhs.fb;
        const auto& cs = problem.rhs.fs;
        const auto& cu = problem.rhs.fu;

        // Auxiliary alias to solution data members
        auto& xb = solution.ub;
        auto& xs = solution.us;
        auto& xu = solution.uu;
        auto& y  = solution.v;
        auto& zb = solution.wb;
        auto& zs = solution.ws;
        auto& zu = solution.wu;

        // Auxiliary variables
        const Index nb = Gb.rows();
        const Index ns = Gs.rows();
        const Index nu = Gu.rows();
        const Index pb = Eb.rows();
        const Index ps = Es.rows();
        const Index pu = Eu.rows();

        // Compute the LU factorization of the canonical saddle point problem
        if(ns) invGsEs = Gs; if(ps) invGsEs -= Es; invGsEs = inv(invGsEs);
        if(pu) invEuGu = Eu; if(nu) invEuGu -= Gu; invEuGu = inv(invEuGu);

        if(nu) invEucu = cu; if(pu) invEucu.array() /= Eu.array();

        if(nb) Lb = Gb; if(pb) Lb -= Eb; Lb.array() /= Bb.array();
        if(ns) Ts = diag(-Lb) * Bs;
        if(nu) Tu = diag( Lb) * Bu;
        if(ns) Ls = Ts * diag(invGsEs);
        if(nb) Tb = diag(Bb); if(ns) Tb -= Ls * tr(Bs);
        if(nu) Lu = Tu * diag(invEuGu);
        if(nb) Ub = Tb; if(nu) Ub -= Lu * tr(Bu);

        if(nb) ub = b; if(nu) ub -= Bu * invEucu;
        if(ns) us = as; if(ps) us -= cs;
        if(nu) uu = au - Gu % invEucu;
        if(nb) vb = ab; if(pb) vb -= cb;

        // Compute the solution
        if(nb) r = vb - Lb%ub; if(ns) r -= Ls*us; if(nu) r -= Lu*uu;
        if(nb) y  = Ub.lu().solve(r);
        if(pu) zu = (uu - tr(Bu)*y) % invEuGu;
        if(ns) xs = (us - tr(Bs)*y) % invGsEs;
        if(nb) xb = ub; if(ns) xb -= Bs*xs; if(pu) xb += Bu*zu; if(nb) xb.array() /= Bb.array();
        if(pb) zb = cb/Eb - xb;
        if(ps) zs = cs/Es - xs;
        if(pu) xu = cu/Eu - zu; else xu = cu;
    }
};

struct SaddlePointSolver::Impl
{
    SaddlePointSolverDiagonal spsd;

    auto solve(const SaddlePointProblemCanonical& problem, SaddlePointVectorCanonical& solution) -> void
    {
        spsd.solve(problem, solution);
    }

    auto solve(const SaddlePointProblem& problem, SaddlePointVector& solution) -> void
    {
        const Index m = problem.lhs.A.rows();
        const Index n = problem.lhs.A.cols();

        Assert(m < n, "", "");

        const auto& X = problem.lhs.X;
        const auto& Z = problem.lhs.Z;
        const auto& H = problem.lhs.H;
        const auto& A = problem.lhs.A;
        const auto& a = problem.rhs.a;
        const auto& b = problem.rhs.b;
        const auto& c = problem.rhs.c;

        CanonicalMatrix C;
        C.compute(A);
        C.update(X);

        const Matrix& S = C.S();
        const Matrix& R = C.R();
        const Indices& ibasic = C.ibasic();
        const Indices& inonbasic = C.inonbasic();

        const Vector Xb = rows(X, ibasic);
        const Vector Zb = rows(Z, ibasic);
        const Vector Hb = rows(H, ibasic);
        const Vector ab = rows(a, ibasic);
        const Vector cb = rows(c, ibasic);

        auto Xn = rows(X, inonbasic);
        auto Zn = rows(Z, inonbasic);
        auto Hn = rows(H, inonbasic);
        auto an = rows(a, inonbasic);
        auto cn = rows(c, inonbasic);

        const Vector G =  X % H % X;
        const Vector E = -X % Z;
        const Vector d =  X % a;
        const Vector f = -c;

        auto Gn = rows(G, inonbasic);
        Indices istable, iunstable;
        istable.reserve(n);
        iunstable.reserve(n);

        for(Index i = 0; i < n - m; ++i)
        {
            const double j = inonbasic[i];
            const double Gj = std::abs(X[j] * H[j] * X[j]);
            const double Ej = std::abs(X[j] * Z[j]);
            if(Gj > Ej) istable.push_back(i);
            else iunstable.push_back(i);
        }

        const Matrix Bn = S * diag(Xn);

        const Vector Xs = rows(X, istable);
        const Vector Zs = rows(Z, istable);
        const Vector Hs = rows(H, istable);
        const Vector Xu = rows(X, iunstable);
        const Vector Zu = rows(Z, iunstable);
        const Vector Hu = rows(H, iunstable);

        SaddlePointProblemCanonical cproblem;

        {Index i = 0; for(Index j : ibasic)
        {
            cproblem.lhs.Bb[i] = X[j];
            cproblem.lhs.Gb[i] = G[j];
            cproblem.lhs.Eb[i] = E[j];
            cproblem.rhs.db[i] = d[j];
            cproblem.rhs.fb[i] = f[j];
            ++i;
        }}

        {Index i = 0; for(Index j : istable)
        {
            cproblem.lhs.Bs.col(i) = Bn.col(j);
            cproblem.lhs.Gs[i]     = G[inonbasic[j]];
            cproblem.lhs.Es[i]     = E[inonbasic[j]];
            cproblem.rhs.ds[i]     = d[inonbasic[j]];
            cproblem.rhs.fs[i]     = f[inonbasic[j]];
            ++i;
        }}

        {Index i = 0; for(Index j : iunstable)
        {
            cproblem.lhs.Bu.col(i) = Bn.col(j);
            cproblem.lhs.Gu[i]     = G[inonbasic[j]];
            cproblem.lhs.Eu[i]     = E[inonbasic[j]];
            cproblem.rhs.du[i]     = d[inonbasic[j]];
            cproblem.rhs.fu[i]     = f[inonbasic[j]];
            ++i;
        }}

        cproblem.rhs.e.noalias()  = R * b;

        SaddlePointVectorCanonical csolution;

        solve(cproblem, csolution);

        auto& xb = rows(solution.x, ibasic);
        auto& xs = rows(solution.x, istable);
        auto& xu = rows(solution.x, iunstable);
        auto& y  = solution.y;
        auto& zb = rows(solution.z, ibasic);
        auto& zs = rows(solution.z, istable);
        auto& zu = rows(solution.z, iunstable);

        xb =  Xb % csolution.ub;
        xs =  Xs % csolution.us;
        xu =  Xu % csolution.uu;
        y  = -tr(R) * csolution.v;
        zb =  Zb % csolution.wb;
        zs =  Zs % csolution.ws;
        zu =  Zu % csolution.wu;
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

auto SaddlePointSolver::constantA(bool isconst) -> void
{
//    pimpl->constA = isconst; TODO implement this functionality
}

auto SaddlePointSolver::solve(const SaddlePointProblem& problem, SaddlePointVector& solution) -> void
{
    assert(0 && "Not implemented yet!");
}

auto SaddlePointSolver::solve(const SaddlePointProblemCanonical& problem, SaddlePointVectorCanonical& solution) -> void
{
    pimpl->solve(problem, solution);
}

} // namespace Optima
