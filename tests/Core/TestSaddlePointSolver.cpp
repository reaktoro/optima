// Optima is a C++ library for numerical sol of linear and nonlinear programing problems.
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

#include <doctest/doctest.hpp>






#include <Optima/Math/Canonicalizer.hpp>





// C++ includes
#include <iostream>

// Eigenx includes
#include <Eigenx/Core.hpp>
#include <Eigenx/LU.hpp>
using namespace Eigen;

// Optima includes
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Core/SaddlePointSolver.hpp>
using namespace Optima;




//TEST_CASE("Testing SaddlePointSolver with RangespaceDiagonal method.")
//{
//    Index m = 10;
//    Index n = 60;
//    Index t = m + n;
//
//    VectorXd expected = linspace(t, 1, t);
//
//    MatrixXd A = random(m, n);
//    MatrixXd H = diag(random(n));
//
//    SaddlePointMatrix lhs(H, A);
//
//    VectorXd r = lhs * expected;
//    VectorXd s(t);
//
//    SaddlePointVector rhs(r, n, m);
//    SaddlePointSolution sol(s, n, m);
//
//    SaddlePointSolver solver;
//    solver.setMethodRangespaceDiagonal();
//    solver.canonicalize(A);
//    solver.decompose(lhs);
//    solver.solve(rhs, sol);
//
//    CHECK_EQ(solver.method(), SaddlePointMethod::RangespaceDiagonal);
//    CHECK(s.isApprox(expected));
//}
//
//TEST_CASE("Testing SaddlePointSolver with RangespaceDiagonal method and fixed variables.")
//{
//    Index m = 10;
//    Index n = 60;
//    Index t = m + n;
//
//    VectorXd expected = linspace(t, 1, t);
//
//    MatrixXd A = random(m, n);
//    MatrixXd H = diag(random(n));
//    Indices ifixed = {0, 10, 20, 30, 40, 50};
//
//    SaddlePointMatrix lhs(H, A, ifixed);
//
//    VectorXd r = lhs * expected;
//    VectorXd s(t);
//
//    SaddlePointVector rhs(r, n, m);
//    SaddlePointSolution sol(s, n, m);
//
//    SaddlePointSolver solver;
//    solver.setMethodRangespaceDiagonal();
//    solver.canonicalize(A);
//    solver.decompose(lhs);
//    solver.solve(rhs, sol);
//
//    CHECK_EQ(solver.method(), SaddlePointMethod::RangespaceDiagonal);
//    CHECK(s.isApprox(expected));
//}

TEST_CASE("Testing SaddlePointSolver with RangespaceDiagonal method and fixed variables.")
{
    Index m = 10;
    Index n = 60;
//    Index n = 15;
    Index t = m + n;

    VectorXd expected = linspace(t, 1, t);

    MatrixXd A = random(m, n);
    MatrixXd H = diag(random(n));

    Indices ifixed, inofixed;
    for(Index i = 0; i < n; ++i)
        if(A(0, i) <= 0) { A(0, i) = 0.0; inofixed.push_back(i); }
        else ifixed.push_back(i);
//        if(A(0, i) <= 0) ifixed.push_back(i);

    auto nf = ifixed.size();
    auto nx = n - nf;

    Canonicalizer canonicalizer(A);

    const auto& R = canonicalizer.R();
    const auto& Q = canonicalizer.Q();
    const auto& S = canonicalizer.S();

    std::cout << "R = \n" << R << std::endl;
    std::cout << "R*A*Q - C = \n" << R*A*Q - canonicalizer.C() << std::endl;

    VectorXd weights = ones(n);
//    rows(weights, ifixed) = -linspace(ifixed.size(), 1, ifixed.size());
    rows(weights, ifixed).fill(0.0);

//    std::cout << "A = \n" << A << std::endl;
//    std::cout << "A(nofixed) = \n" << cols(A, inofixed) << std::endl;

    canonicalizer.update(weights);

    std::cout << "R = \n" << R << std::endl;

    std::cout << "R*A*Q - C = \n" << R*A*Q - canonicalizer.C() << std::endl;

    auto wb = rows(weights, canonicalizer.ibasic());
    auto wn = rows(weights, canonicalizer.inonbasic());

    std::cout << "wb = \n" << tr(wb) << std::endl;
    std::cout << "wn = \n" << tr(wn) << std::endl;

    MatrixXd M(m, nx); M << identity(m,m), S.leftCols(nx-m);

    auto lu = M.fullPivLu();

    std::cout << "M = \n" << M << std::endl;
    std::cout << "rank(M) = " << lu.rank() << std::endl;

    SaddlePointMatrix lhs(H, A, ifixed);

    VectorXd r = lhs * expected;
    VectorXd s(t);

    SaddlePointVector rhs(r, n, m);
    SaddlePointSolution sol(s, n, m);

    SaddlePointSolver solver;
    solver.setMethodRangespaceDiagonal();
    solver.canonicalize(A);
    solver.decompose(lhs);
    solver.solve(rhs, sol);

    CHECK_EQ(solver.method(), SaddlePointMethod::RangespaceDiagonal);

    double error = (lhs.matrix() * sol.vector() - rhs.vector()).norm()/rhs.vector().norm();
    CHECK(approx(error) == 0.0);
}

//TEST_CASE("Testing SaddlePointSolver with other methods.")
//{
//    Index m = 10;
//    Index n = 60;
//    Index t = m + n;
//
//    VectorXd expected = linspace(t, 1, t);
//
//    MatrixXd A = random(m, n);
//    MatrixXd H = random(n, n);
//
//    SaddlePointMatrix lhs(H, A);
//
//    VectorXd r = lhs * expected;
//    VectorXd s(t);
//
//    SaddlePointVector rhs(r, n, m);
//    SaddlePointSolution sol(s, n, m);
//
//    SUBCASE("When method is SaddlePointMethod::Nullspace")
//    {
//        SaddlePointSolver solver;
//        solver.setMethodNullspace();
//        solver.canonicalize(A);
//        solver.decompose(lhs);
//        solver.solve(rhs, sol);
//        CHECK_EQ(solver.method(), SaddlePointMethod::Nullspace);
//        CHECK(s.isApprox(expected));
//    }
//
//    SUBCASE("When method is SaddlePointMethod::PartialPivLU")
//    {
//        SaddlePointSolver solver;
//        solver.setMethodPartialPivLU();
//        solver.canonicalize(A);
//        solver.decompose(lhs);
//        solver.solve(rhs, sol);
//        CHECK_EQ(solver.method(), SaddlePointMethod::PartialPivLU);
//        CHECK(s.isApprox(expected));
//    }
//
//    SUBCASE("When method is SaddlePointMethod::FullPivLU")
//    {
//        SaddlePointSolver solver;
//        solver.setMethodFullPivLU();
//        solver.canonicalize(A);
//        solver.decompose(lhs);
//        solver.solve(rhs, sol);
//        CHECK_EQ(solver.method(), SaddlePointMethod::FullPivLU);
//        CHECK(s.isApprox(expected));
//    }
//}

