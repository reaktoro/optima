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

// C++ includes
#include <iostream>

// Eigenx includes
#include <Eigenx/Core.hpp>
#include <Eigenx/LU.hpp>
using namespace Eigen;

// Optima includes
#include <Optima/Core/HessianMatrix.hpp>
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointSolver.hpp>
using namespace Optima;

Index samples = 10;

TEST_CASE("Testing the solution of a saddle point problem with diagonal Hessian")
{
    Index m = 10;
    Index n = 60;
    Index t = m + n;

    VectorXd expected = linspace(t, 1, t);

    MatrixXd A = random(m, n);
    VectorXd H = random(n);

    SaddlePointMatrix lhs(H, A);

    MatrixXd M = lhs.matrix();
    VectorXd r = M * expected;
    VectorXd s(t);

    SaddlePointVector rhs(r, n, m);
    SaddlePointSolution sol(s, n, m);

    SaddlePointResult res1, res2, res3, res;

    for(Index i = 0; i < samples; ++i)
    {
        SaddlePointSolver solver;
        res1 += solver.canonicalize(lhs);
        res2 += solver.decompose(lhs);
        res3 += solver.solve(rhs, sol);

        res += res1 + res2 + res3;

        CHECK(s.isApprox(expected));
    }

    double timesps1 = res1.time()/samples;
    double timesps2 = res2.time()/samples;
    double timesps3 = res3.time()/samples;
    double timesps  = res.time()/samples;

    double timelu1 = 0.0, timelu2 = 0.0, timelu = 0.0;
    VectorXd slu;

    for(Index i = 0; i < samples; ++i)
    {
        Time begin = time();
        PartialPivLU<MatrixXd> lu(M);
        timelu1 += elapsed(begin);
        begin = time();
        slu = lu.solve(r);
        timelu2 += elapsed(begin);
        timelu  += timelu1 + timelu2;
    }

    timelu1 /= samples; timelu2 /= samples; timelu /= samples;

    std::cout << std::endl;
    std::cout << "Error(SaddlePointSolver): " << norminf(s - expected) << std::endl;
    std::cout << "Error(PartialPivLU):      " << norminf(slu - expected) << std::endl;
    std::cout << std::endl;
    std::cout << "Time(SaddlePointSolver::canonicalize): " << timesps1 << std::endl;
    std::cout << "Time(SaddlePointSolver::decompose):    " << timesps2 << std::endl;
    std::cout << "Time(SaddlePointSolver::solve):        " << timesps3 << std::endl;
    std::cout << "Time(SaddlePointSolver::all):          " << timesps  << std::endl;
    std::cout << std::endl;
    std::cout << "Time(PartialPivLU::decompose): " << timelu1 << std::endl;
    std::cout << "Time(PartialPivLU::solve):     " << timelu2 << std::endl;
    std::cout << std::endl;
    std::cout << "Speedup(canonicalize+decompose): " << timelu1/(timesps1 + timesps2) << std::endl;
    std::cout << "Speedup(decompose):              " << timelu1/timesps2 << std::endl;
    std::cout << "Speedup(solve):                  " << timelu2/timesps3 << std::endl;
    std::cout << "Speedup(decompose+solve):        " << timelu/(timesps2 + timesps3) << std::endl;
//    std::cout << "Speedup(total):                  " << timelu/timesps << std::endl;
}

//
//TEST_CASE("Testing the solution of a saddle point problem with diagonal Hessian")
//{
//    Index m = 2;
//    Index n = 4;
//    Index t = 2*n + m;
//
//    SaddlePointMatrix lhs;
//    lhs.A = random(m, n);
//
//    Indices fixed = {0};
//
//    SaddlePointSolver solver;
//    solver.fix(fixed);
//    solver.canonicalize(lhs);
//
//    SaddlePointVector sol;
//    sol.x = zeros(n);
//    sol.y = zeros(m);
//
//    Vector expected = linspace(t, 1, t);
//    for(Index i : fixed)
//        expected[i] = expected[i+n+m] = 0.0;
////    rows(expected, ignored).fill(0.0);
//
//    SUBCASE("Hessian matrix is zero.")
//    {
//        lhs.H = zeros(n);
//
//        std::cout << "------------------------------" << std::endl;
//        std::cout << "Case 1: Hessian matrix is zero" << std::endl;
//        std::cout << "------------------------------" << std::endl;
//
//        std::cout << std::left << std::setw(5) << "k";
//        std::cout << std::left << std::setw(15) << "Error";
//        std::cout << std::left << std::setw(15) << "Error (LU)";
//        std::cout << std::endl;
//        for(Index k = 0; k < 10; ++k)
//        {
//            lhs.X.head(n - m) *= 1e-5;
//            lhs.Z.head(n - m) *= 10.0;
//
//            Matrix A = convert(lhs, fixed);
//            Vector b = A * expected;
//            Vector x = A.lu().solve(b);
//
//            SaddlePointVector rhs;
//            rhs.x = b.head(n);
//            rhs.y = b.segment(n, m);
//            rhs.z = b.tail(n);
//
//            REQUIRE(lhs.valid());
//            REQUIRE(rhs.valid());
//
//            solver.decompose(lhs);
//            solver.solve(rhs, sol);
//
//            auto actual = sol.convert();
//
//            CHECK(actual.isApprox(x));
//
//            std::cout << std::left << std::setw(5) << k;
//            std::cout << std::left << std::setw(15) << norm(actual - expected);
//            std::cout << std::left << std::setw(15) << norm(x - expected);
//            std::cout << std::endl;
//        }
//    }

//    SUBCASE("Hessian matrix is not zero.")
//    {
//        lhs.H = random(n);
//
//        std::cout << "----------------------------------" << std::endl;
//        std::cout << "Case 2: Hessian matrix is not zero" << std::endl;
//        std::cout << "----------------------------------" << std::endl;
//
//        std::cout << std::left << std::setw(5) << "k";
//        std::cout << std::left << std::setw(15) << "Error";
//        std::cout << std::left << std::setw(15) << "Error (LU)";
//        std::cout << std::endl;
//        for(Index k = 0; k < 10; ++k)
//        {
//            lhs.X.head(n - m) *= 1e-5;
//            lhs.Z.head(n - m) *= 10.0;
//
//            Matrix A = lhs.convert();
//            Vector b = A * expected;
//            Vector x = A.lu().solve(b);
//
//            SaddlePointVector rhs;
//            rhs.x = b.head(n);
//            rhs.y = b.segment(n, m);
//            rhs.z = b.tail(n);
//
//            REQUIRE(lhs.valid());
//            REQUIRE(rhs.valid());
//
//            solver.decompose(lhs);
//            solver.solve(rhs, sol);
//
//            auto actual = sol.convert();
//
//            CHECK(actual.isApprox(x));
//
//            std::cout << std::left << std::setw(5) << k;
//            std::cout << std::left << std::setw(15) << norm(actual - expected);
//            std::cout << std::left << std::setw(15) << norm(x - expected);
//            std::cout << std::endl;
//        }
//    }
//}

