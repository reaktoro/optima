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

// Optima includes
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointSolver.hpp>
#include <Optima/Math/Eigen/LU>
using namespace Optima;

Index samples = 10;

TEST_CASE("Testing the solution of a saddle point problem with diagonal Hessian")
{
    Index m = 10;
    Index n = 60;

    SaddlePointMatrix lhs;
    lhs.A = random(m, n);
    lhs.H = random(n);

    SaddlePointVector sol;
    sol.x = linspace(n, 1, n);
    sol.y = linspace(m, 1, m);

    SaddlePointVector rhs = lhs * sol;

    SaddlePointVector actualsol;

    SaddlePointResult res1, res2, res3, res;

    for(Index i = 0; i < samples; ++i)
    {
        SaddlePointSolver solver;
        res1 += solver.canonicalize(lhs);
        res2 += solver.decompose(lhs);
        res3 += solver.solve(rhs, actualsol);

        res += res1 + res2 + res3;

        CHECK(sol.x.isApprox(actualsol.x));
        CHECK(sol.y.isApprox(actualsol.y));
    }

    double timesps1 = res1.time()/samples;
    double timesps2 = res2.time()/samples;
    double timesps3 = res3.time()/samples;
    double timesps  = res.time()/samples;

    Matrix A = lhs.matrix();
    Vector b = rhs.vector();
    Vector x = sol.vector();

    double timelu1 = 0.0, timelu2 = 0.0, timelu = 0.0;
    Vector actualx;

    for(Index i = 0; i < samples; ++i)
    {
        Time begin = time();
        Eigen::PartialPivLU<Matrix> lu(A);
        timelu1 += elapsed(begin);
        begin = time();
        actualx = lu.solve(b);
        timelu2 += elapsed(begin);
        timelu  += timelu1 + timelu2;
    }

    timelu1 /= samples; timelu2 /= samples; timelu /= samples;

    std::cout << std::endl;
    std::cout << "Error(SaddlePointSolver): " << norminf(actualsol.vector() - sol.vector()) << std::endl;
    std::cout << "Error(PartialPivLU):      " << norminf(actualx - x) << std::endl;
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

