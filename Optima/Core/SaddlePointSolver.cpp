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

// Eigen includes
#include <Optima/Math/Eigen/LU>

namespace Optima {

struct SaddlePointSolver::Impl
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
        const auto& ab = problem.rhs.ab;
        const auto& as = problem.rhs.as;
        const auto& au = problem.rhs.au;
        const auto& b  = problem.rhs.b;
        const auto& cb = problem.rhs.cb;
        const auto& cs = problem.rhs.cs;
        const auto& cu = problem.rhs.cu;

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
    pimpl->constA = isconst;
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
