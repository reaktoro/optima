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


// todo remove
#include <iostream>

// Eigen includes
#include <Optima/Math/Eigen/LU>

namespace Optima {

auto solver(const SaddlePointProblemCanonical& problem, SaddlePointSolutionCanonical& solution) -> void
{
    // Auxiliary alias to problem data members
    const auto& Gb   = problem.Gb;
    const auto& Gs   = problem.Gs;
    const auto& Gu   = problem.Gu;
    const auto& Bb   = problem.Bb;
    const auto& Bs   = problem.Bs;
    const auto& Bu   = problem.Bu;
    const auto& Eb   = problem.Eb;
    const auto& Es   = problem.Es;
    const auto& Eu   = problem.Eu;
    const auto& ab   = problem.ab;
    const auto& as   = problem.as;
    const auto& au   = problem.au;
    const auto& b    = problem.b;
    const auto& cb   = problem.cb;
    const auto& cs   = problem.cs;
    const auto& cu   = problem.cu;

    // Auxiliary alias to solution data members
    auto& xb = solution.xb;
    auto& xs = solution.xs;
    auto& xu = solution.xu;
    auto& y  = solution.y;
    auto& zb = solution.zb;
    auto& zs = solution.zs;
    auto& zu = solution.zu;

    // Auxiliary variables
    const auto nb = Gb.rows();
    const auto ns = Gs.rows();
    const auto nu = Gu.rows();

    // Compute the LU factorization of the canonical saddle point problem
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

    if(nb) Lb = (Gb - Eb)/Bb; // TODO put these aux vectors/matrices as data members of a solver class
    if(ns) Ts = diag(-Lb) * Bs;
    if(nu) Tu = diag( Lb) * Bu;
    if(ns) Ls = Ts * diag(inv(Gs - Es));
    if(nb) Tb = diag(Bb); if(ns) Tb -= Ls * tr(Bs);
    if(nu) Lu = Tu * diag(inv(Eu - Gu));
    if(nb) Ub = Tb; if(nu) Ub -= Lu * tr(Bu);

    if(nb) ub = b; if(nu) ub -= Bu * (cu/Eu);
    if(ns) us = as - cs;
    if(nu) uu = au - Gu % (cu/Eu);
    if(nb) vb = ab - cb;

    // Compute the solution
    if(nb) r = vb - Lb%ub; if(ns) r -= Ls*us; if(nu) r -= Lu*uu;
    if(nb) y  = Ub.lu().solve(r); // TODO create a PartialPivLU instance (check for positive definiteness and LLT solver too!)
    if(nu) zu = (uu - tr(Bu)*y)/(Eu - Gu);
    if(ns) xs = (us - tr(Bs)*y)/(Gs - Es);
    if(nb) xb = ub; if(ns) xb -= Bs*xs; if(nu) xb += Bu*zu; if(nb) xb.array() /= Bb.array();
    if(nb) zb = cb/Eb - xb;
    if(ns) zs = cs/Es - xs;
    if(nu) xu = cu/Eu - zu;
}

} // namespace Optima
