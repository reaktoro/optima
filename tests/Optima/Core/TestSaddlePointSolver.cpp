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

// Optima includes
#include <Optima/Optima.hpp>
#include <Optima/Math/Eigen/LU>
using namespace Optima;

TEST_CASE("Testing the solution of a saddle point problem with diagonal Hessian")
{
    Index m = 10;
    Index n = 15;
    Index t = 2*n + m;

    SaddlePointMatrix lhs;
    lhs.A = random(m, n);
    lhs.X = abs(random(n));
    lhs.Z = abs(random(n));


    SaddlePointSolver solver;
    solver.fixed({0}, {2});
    solver.canonicalize(lhs);

    SaddlePointVector sol;

    Vector expected = linspace(t, 1, t);
    expected[0] = 2;
    expected[n + m] = 0;

    SUBCASE("Hessian matrix is zero.")
    {
        lhs.H = zeros(n);

        std::cout << "------------------------------" << std::endl;
        std::cout << "Case 1: Hessian matrix is zero" << std::endl;
        std::cout << "------------------------------" << std::endl;

        std::cout << std::left << std::setw(5) << "k";
        std::cout << std::left << std::setw(15) << "Error";
        std::cout << std::left << std::setw(15) << "Error (LU)";
        std::cout << std::endl;
        for(Index k = 0; k < 10; ++k)
        {
            lhs.X.head(n - m) *= 1e-5;
            lhs.Z.head(n - m) *= 10.0;

            Matrix A = lhs.convert();
            A.row(0).fill(0.0);
            A(0,0) = 1.0;
            A(0,0) = 1.0;
            A(n+m,0) = 0.0; // Z
            A(n+m,n+m) = 1.0; // X

            Vector b = A * expected;

            Vector x = A.lu().solve(b);

            SaddlePointVector rhs;
            rhs.x = b.head(n);
            rhs.y = b.segment(n, m);
            rhs.z = b.tail(n);

            REQUIRE(lhs.valid());
            REQUIRE(rhs.valid());

            solver.decompose(lhs);
            solver.solve(rhs, sol);

            auto actual = sol.convert();

            CHECK(actual.isApprox(x));

            std::cout << std::left << std::setw(5) << k;
            std::cout << std::left << std::setw(15) << norm(actual - expected);
            std::cout << std::left << std::setw(15) << norm(x - expected);
            std::cout << std::endl;
        }
    }

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
}

