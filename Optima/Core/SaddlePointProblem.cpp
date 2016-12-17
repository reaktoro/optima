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

struct SaddlePointProblem::Impl
{
    /// The left-hand side coefficient matrix of the saddle point problem in canonical and scaled form.
    SaddlePointMatrixCanonical clhs;

    /// The right-hand side vector of the saddle point problem in canonical and scaled form.
    SaddlePointVector crhs;

    /// The canonicalizer of the Jacobian matrix `A`.
    Canonicalizer canonicalizer;

    /// The auxiliary data to calculate the scaling of the saddle point problem.
    Vector X, Z;

    /// The threshold parameter used to detect unstable variables.
    double eps = 1e-15;

    /// Canonicalize the Jacobian matrix `A` of the saddle point problem.
    auto canonicalize(const SaddlePointMatrix& lhs) -> void
    {
        // Compute the canonical form of matrix A
        canonicalizer.compute(lhs.A);
    }

    /// Update the left-hand side coefficient matrix of the saddle point problem.
    auto lhs(const SaddlePointMatrix& lhs) -> void
    {
        // Alias to members of the saddle point matrix
        const auto& H = lhs.H;

        // Alias to members of the canonical saddle point matrix
        auto& G  = clhs.G;
        auto& Bb = clhs.Bb;
        auto& Bn = clhs.Bn;
        auto& E  = clhs.E;
        auto& nb = clhs.nb;
        auto& nn = clhs.nn;
        auto& ns = clhs.ns;
        auto& nu = clhs.nu;

        // Update vectors X and Z
        X.noalias() = lhs.X;
        Z.noalias() = lhs.Z;

        // Update the canonical form and the variables order with currect X values
        canonicalizer.update(X);

        // The number of rows and columns of the canonical form of A
        const Index m = canonicalizer.rows();
        const Index n = canonicalizer.cols();

        // Set the number of basic and non-basic variables
        nb = m;
        nn = n - nb;

        // Compute the scaled matrices G and E
        G.noalias() =  X % H % X;
        E.noalias() = -X % Z;

        // Get the updated ordering of the variables
        const auto& Q = canonicalizer.Q();
        const auto& S = canonicalizer.S();

//        G.noalias() = Q.transpose() * G * Q; dense case
        X.noalias() = X * Q;
        Z.noalias() = Z * Q;

        auto Xb = X.head(nb);
        auto Xn = X.tail(nn);

        G.noalias()  = G * Q;
        E.noalias()  = E * Q;
        Bb.noalias() = Xb;
        Bn.noalias() = S * Xn;

        // Find the number of non-basic stable variables.
        const double minXb = std::abs(Xb.minCoeff());
        const double threshold = eps * minXb;

        // Iterate over all non-basic variables and determine the first that is considered "too small"
        for(ns = 0; ns < nn; ++ns)
            if(std::abs(Xn[ns]) < threshold)
                break;

        // Set the number of non-basic unstable variables
        nu = nn - ns;
    }

    /// Update the right-hand side vector of the saddle point problem.
    auto rhs(const SaddlePointVector& rhs) -> void;

};


//
//auto SaddlePointProblemCanonical::compute(const SaddlePointProblem& problem) -> void
//{
//    const auto& X = problem.lhs.X;
//    const auto& Z = problem.lhs.Z;
//    const auto& H = problem.lhs.H;
//    const auto& A = problem.lhs.A;
//    const auto& a = problem.rhs.x;
//    const auto& b = problem.rhs.y;
//    const auto& c = problem.rhs.z;
//
//    canonicalizer.compute(A);
//    canonicalizer.update(X);
//
//    const auto& S = canonicalizer.S();
//    const auto& R = canonicalizer.R();
//
//    ibasic = canonicalizer.ibasic();
//    inonbasic = canonicalizer.inonbasic();
//
//    istable.reserve(inonbasic.size());
//    iunstable.reserve(inonbasic.size());
//
//    istable.clear();
//    iunstable.clear();
//    for(Index i = 0; i < inonbasic.size(); ++i)
//    {
//        const double j = inonbasic[i];
//        const double Xj = X[j];
//        const double Zj = Z[j];
//        if(std::abs(Xj) > std::abs(Zj))
//            istable.push_back(i);
//        else iunstable.push_back(i);
//    }
//
//    // Compute the scaled saddle point problem
//    G =  X % H % X;
//    E = -X % Z;
//    r =  X % a;
//    t = -c;
//
//    auto Xb = rows(X, ibasic);
//    auto Xn = rows(X, inonbasic);
//
//    lhs.Gb.noalias() = rows(G, ibasic);
//    lhs.Gs.noalias() = rows(rows(G, inonbasic), istable);
//    lhs.Gu.noalias() = rows(rows(G, inonbasic), iunstable);
//
//    lhs.Bb.noalias() = Xb;
//
//    lhs.Bs.conservativeResize(ibasic.size(), istable.size());
//    lhs.Bu.conservativeResize(ibasic.size(), iunstable.size());
//
//    for(Index i = 0; i < istable.size(); ++i)
//        lhs.Bs.col(i) = S.col(istable[i]) * Xn[istable[i]];
//    for(Index i = 0; i < iunstable.size(); ++i)
//        lhs.Bu.col(i) = S.col(iunstable[i]) * Xn[iunstable[i]];
//
//    lhs.Eb.noalias() = rows(E, ibasic);
//    lhs.Es.noalias() = rows(rows(E, inonbasic), istable);
//    lhs.Eu.noalias() = rows(rows(E, inonbasic), iunstable);
//
//    rhs.xb = rows(r, ibasic);
//    rhs.xs = rows(rows(r, inonbasic), istable);
//    rhs.xu = rows(rows(r, inonbasic), iunstable);
//    rhs.y  = R * b;
//    rhs.zb = rows(t, ibasic);
//    rhs.zs = rows(rows(t, inonbasic), istable);
//    rhs.zu = rows(rows(t, inonbasic), iunstable);
//}
//
//auto SaddlePointProblemCanonical::update(const Vector& weights) -> void
//{
//
//}

} // namespace Optima
