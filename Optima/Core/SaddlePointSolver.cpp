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
        const auto& ab = problem.rhs.rb;
        const auto& as = problem.rhs.rs;
        const auto& au = problem.rhs.ru;
        const auto& b  = problem.rhs.s;
        const auto& cb = problem.rhs.tb;
        const auto& cs = problem.rhs.ts;
        const auto& cu = problem.rhs.tu;

        // Auxiliary alias to solution data members
        auto& xb = solution.xb;
        auto& xs = solution.xs;
        auto& xu = solution.xu;
        auto& y  = solution.y;
        auto& zb = solution.zb;
        auto& zs = solution.zs;
        auto& zu = solution.zu;

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

        const Vector G =  X % H % X;
        const Vector E = -X % Z;
        const Vector r =  X % a;
        const Vector t = -c;

        Indices istable, iunstable;
        istable.reserve(n);
        iunstable.reserve(n);

        for(Index i = 0; i < inonbasic.size(); ++i)
        {
            const double j = inonbasic[i];
            if(std::abs(G[j]) > std::abs(E[j]))
                istable.push_back(i);
            else iunstable.push_back(i);
        }

        SaddlePointProblemCanonical cproblem;
        auto& Gb = cproblem.lhs.Gb;
        auto& Gs = cproblem.lhs.Gs;
        auto& Gu = cproblem.lhs.Gu;
        auto& Bb = cproblem.lhs.Bb;
        auto& Bs = cproblem.lhs.Bs;
        auto& Bu = cproblem.lhs.Bu;
        auto& Eb = cproblem.lhs.Eb;
        auto& Es = cproblem.lhs.Es;
        auto& Eu = cproblem.lhs.Eu;
        auto& rb = cproblem.rhs.rb;
        auto& rs = cproblem.rhs.rs;
        auto& ru = cproblem.rhs.ru;
        auto&  s = cproblem.rhs.s ;
        auto& tb = cproblem.rhs.tb;
        auto& ts = cproblem.rhs.ts;
        auto& tu = cproblem.rhs.tu;

        auto Xb = rows(X, ibasic);
        auto Xn = rows(X, inonbasic);

        auto Zb = rows(Z, ibasic);
        auto Zn = rows(Z, inonbasic);

        auto Xs = rows(Xn, istable);
        auto Xu = rows(Xn, iunstable);

        auto Zs = rows(Zn, istable);
        auto Zu = rows(Zn, iunstable);

        auto Ss = cols(S, istable);
        auto Su = cols(S, iunstable);

        Gb.noalias() = rows(G, ibasic);
        Gs.noalias() = rows(rows(G, inonbasic), istable);
        Gu.noalias() = rows(rows(G, inonbasic), iunstable);

        Bb.noalias() = Xb;

        Bs.conservativeResize(ibasic.size(), istable.size());
        Bu.conservativeResize(ibasic.size(), iunstable.size());

        for(Index i = 0; i < istable.size(); ++i)
            Bs.col(i) = S.col(istable[i]) * Xn[istable[i]];
        for(Index i = 0; i < iunstable.size(); ++i)
            Bu.col(i) = S.col(iunstable[i]) * Xn[iunstable[i]];

        Eb.noalias() = rows(E, ibasic);
        Es.noalias() = rows(rows(E, inonbasic), istable);
        Eu.noalias() = rows(rows(E, inonbasic), iunstable);

        rb = rows(r, ibasic);
        rs = rows(rows(r, inonbasic), istable);
        ru = rows(rows(r, inonbasic), iunstable);
        s  = R * b;
        tb = rows(t, ibasic);
        ts = rows(rows(t, inonbasic), istable);
        tu = rows(rows(t, inonbasic), iunstable);

        SaddlePointVectorCanonical csolution;

        Vector tmp(cproblem.lhs.rows());
        Vector dx(n), dy(m), dz(n);
        dx << inv(Xb), inv(Xs), inv(Xu);
        dy = -tr(C.Rinv()) * ones(m);
        dz << inv(Zb), inv(Zs), inv(Zu);
        tmp << dx, dy, dz;

        solve(cproblem, csolution);

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

        xb =  Xb % csolution.xb;
        xs =  Xs % csolution.xs;
        xu =  Xu % csolution.xu;
        y.noalias()  = -tr(R) * csolution.y;
        zb = Zb % csolution.zb;
        zs = Zs % csolution.zs;
        zu = Zu % csolution.zu;
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
    pimpl->solve(problem, solution);
}

auto SaddlePointSolver::solve(const SaddlePointProblemCanonical& problem, SaddlePointVectorCanonical& solution) -> void
{
    pimpl->solve(problem, solution);
}

} // namespace Optima
