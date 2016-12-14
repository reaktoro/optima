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
#include <Optima/Math/Eigen/LU>
#include <Optima/Math/Eigen/Cholesky>

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
        const auto& ab = problem.rhs.xb;
        const auto& as = problem.rhs.xs;
        const auto& au = problem.rhs.xu;
        const auto& b  = problem.rhs.y;
        const auto& cb = problem.rhs.zb;
        const auto& cs = problem.rhs.zs;
        const auto& cu = problem.rhs.zu;

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
        const auto& a = problem.rhs.x;
        const auto& b = problem.rhs.y;
        const auto& c = problem.rhs.z;

        Canonicalizer C;
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
        auto& rb = cproblem.rhs.xb;
        auto& rs = cproblem.rhs.xs;
        auto& ru = cproblem.rhs.xu;
        auto&  s = cproblem.rhs.y ;
        auto& tb = cproblem.rhs.zb;
        auto& ts = cproblem.rhs.zs;
        auto& tu = cproblem.rhs.zu;

        auto Xb = rows(X, ibasic);
        auto Xn = rows(X, inonbasic);

        auto Zb = rows(Z, ibasic);
        auto Zn = rows(Z, inonbasic);

        auto Xs = rows(Xn, istable);
        auto Xu = rows(Xn, iunstable);

        auto Zs = rows(Zn, istable);
        auto Zu = rows(Zn, iunstable);

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

        Vector dx(n), dy(m), dz(n);
        dx << inv(Xb), inv(Xs), inv(Xu);
        dy = -tr(C.Rinv()) * ones(m);
        dz << inv(Zb), inv(Zs), inv(Zu);

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

auto solve(SaddlePointProblemCanonical& problem, SaddlePointVectorCanonical& solution) -> void
{
    auto& Gb = problem.lhs.Gb;
    auto& Gs = problem.lhs.Gs;
    auto& Gu = problem.lhs.Gu;
    auto& Bb = problem.lhs.Bb;
    auto& Bs = problem.lhs.Bs;
    auto& Bu = problem.lhs.Bu;
    auto& Eb = problem.lhs.Eb;
    auto& Es = problem.lhs.Es;
    auto& Eu = problem.lhs.Eu;
    auto& rb = problem.rhs.xb;
    auto& rs = problem.rhs.xs;
    auto& ru = problem.rhs.xu;
    auto&  s = problem.rhs.y ;
    auto& tb = problem.rhs.zb;
    auto& ts = problem.rhs.zs;
    auto& tu = problem.rhs.zu;

    // Auxiliary alias to solution data members
    auto& xb = solution.xb;
    auto& xs = solution.xs;
    auto& xu = solution.xu;
    auto& y  = solution.y;
    auto& zb = solution.zb;
    auto& zs = solution.zs;
    auto& zu = solution.zu;

    rb.noalias() -= tb;
    rs.noalias() -= ts;
    ru.noalias() -= Gu % (tu/Eu);

    Gb.noalias() -= Eb;
    Gs.noalias() -= Es;
    Gu.noalias() -= Eu;

    tb.noalias() = tb/Eb;
    ts.noalias() = ts/Es;
    tu.noalias() = tu/Eu;

    s.noalias() -= Bu * tu;

    Bs.noalias() = diag(inv(Bb)) * Bs;
    Bu.noalias() = diag(inv(Bb)) * Bu;

    s.noalias() = s/Bb;

    Matrix lhs = diag(inv(Gb));
    lhs.noalias() += Bs*diag(inv(Gs))*tr(Bs);
    lhs.noalias() += Bu*diag(inv(Gu))*tr(Bu);

    Vector rhs = s;
    rhs.noalias() += Bs*diag(inv(Gs))*tr(Bs)*rb;
    rhs.noalias() += Bu*diag(inv(Gu))*tr(Bu)*rb;
    rhs.noalias() -= Bs*(rs/Gs);
    rhs.noalias() -= Bu*(ru/Gu);

    Eigen::LDLT<Matrix> ldlt(lhs);

    xb.noalias() = ldlt.solve(rhs);
    y .noalias() = rb - xb;
    xb.noalias() = xb/Gb;
    xs.noalias() = (rs - tr(Bs)*y)/Gs;
    zu.noalias() = (tr(Bu)*y - ru)/Gu;
    y .noalias() = y/Bb;
    zb.noalias() = tb - xb;
    zs.noalias() = ts - xs;
    xu.noalias() = tu - zu;
}

auto solve2(SaddlePointProblemCanonical& problem, SaddlePointVectorCanonical& solution) -> void
{
    auto& Gb = problem.lhs.Gb;
    auto& Gs = problem.lhs.Gs;
    auto& Gu = problem.lhs.Gu;
    auto& Bb = problem.lhs.Bb;
    auto& Bs = problem.lhs.Bs;
    auto& Bu = problem.lhs.Bu;
    auto& Eb = problem.lhs.Eb;
    auto& Es = problem.lhs.Es;
    auto& Eu = problem.lhs.Eu;
    auto& rb = problem.rhs.xb;
    auto& rs = problem.rhs.xs;
    auto& ru = problem.rhs.xu;
    auto&  s = problem.rhs.y ;
    auto& tb = problem.rhs.zb;
    auto& ts = problem.rhs.zs;
    auto& tu = problem.rhs.zu;

    // Auxiliary alias to solution data members
    auto& xb = solution.xb;
    auto& xs = solution.xs;
    auto& xu = solution.xu;
    auto& y  = solution.y;
    auto& zb = solution.zb;
    auto& zs = solution.zs;
    auto& zu = solution.zu;

    rb -= tb;
    rs -= ts;
    ru -= Gu % (tu/Eu);

    Gb -= Eb;
    Gs -= Es;
    Gu -= Eu;

    tb.noalias() = tb/Eb;
    ts.noalias() = ts/Es;
    tu.noalias() = tu/Eu;

    Eb.noalias() = inv(Eb);
    Es.noalias() = inv(Es);
    Eu.noalias() = inv(Eu);

    s  -= Bu * tu;

    Bb.noalias() = inv(Bb);

    Bs = diag(Bb) * Bs; // check with .noalias()
    Bu = diag(Bb) * Bu; // check with .noalias()

    s = Bb % s; // check with .noalias()

    Gs = inv(Gs); // check with .noalias()
    Gu = inv(Gu); // check with .noalias()

    rs = Gs % rs; // check with .noalias()
    ru = Gu % ru; // check with .noalias()

    Matrix lhs;
    lhs  = diag(Gb);
    lhs += Bs*diag(Gs)*tr(Bs);
    lhs += Bu*diag(Gu)*tr(Bu);

    Vector rhs;
    rhs  = s;
    rhs += Bs*diag(Gs)*tr(Bs)*rb;
    rhs += Bu*diag(Gu)*tr(Bu)*rb;
    rhs -= Bs*rs;
    rhs -= Bu*ru;
    rhs.noalias() = Gb % rhs;

    xb = lhs.ldlt().solve(rhs);
    y  = rb - Gb % xb;
    xs = rs - diag(Gs)*tr(Bs)*y;
    zu = diag(Gu)*tr(Bu)*y - ru;
    y  = Bb % y;
    zb = tb - xb;
    zs = ts - xs;
    xu = tu - zu;
}

} // namespace Optima
