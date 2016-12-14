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

#include "SaddlePointProblem.hpp"

namespace Optima {

auto SaddlePointProblemCanonical::compute(const SaddlePointProblem& problem) -> void
{
    const auto& X = problem.lhs.X;
    const auto& Z = problem.lhs.Z;
    const auto& H = problem.lhs.H;
    const auto& A = problem.lhs.A;
    const auto& a = problem.rhs.x;
    const auto& b = problem.rhs.y;
    const auto& c = problem.rhs.z;

    canonicalizer.compute(A);
    canonicalizer.update(X);

    const auto& S = canonicalizer.S();
    const auto& R = canonicalizer.R();

    ibasic = canonicalizer.ibasic();
    inonbasic = canonicalizer.inonbasic();

    istable.reserve(inonbasic.size());
    iunstable.reserve(inonbasic.size());

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

    // Compute the scaled saddle point problem
    G =  X % H % X;
    E = -X % Z;
    r =  X % a;
    t = -c;

    auto Xb = rows(X, ibasic);
    auto Xn = rows(X, inonbasic);

    lhs.Gb.noalias() = rows(G, ibasic);
    lhs.Gs.noalias() = rows(rows(G, inonbasic), istable);
    lhs.Gu.noalias() = rows(rows(G, inonbasic), iunstable);

    lhs.Bb.noalias() = Xb;

    lhs.Bs.conservativeResize(ibasic.size(), istable.size());
    lhs.Bu.conservativeResize(ibasic.size(), iunstable.size());

    for(Index i = 0; i < istable.size(); ++i)
        lhs.Bs.col(i) = S.col(istable[i]) * Xn[istable[i]];
    for(Index i = 0; i < iunstable.size(); ++i)
        lhs.Bu.col(i) = S.col(iunstable[i]) * Xn[iunstable[i]];

    lhs.Eb.noalias() = rows(E, ibasic);
    lhs.Es.noalias() = rows(rows(E, inonbasic), istable);
    lhs.Eu.noalias() = rows(rows(E, inonbasic), iunstable);

    rhs.xb = rows(r, ibasic);
    rhs.xs = rows(rows(r, inonbasic), istable);
    rhs.xu = rows(rows(r, inonbasic), iunstable);
    rhs.y  = R * b;
    rhs.zb = rows(t, ibasic);
    rhs.zs = rows(rows(t, inonbasic), istable);
    rhs.zu = rows(rows(t, inonbasic), iunstable);
}

auto SaddlePointProblemCanonical::update(const Vector& weights) -> void
{

}

} // namespace Optima
