// Optima is a C++ library for numerical sol of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2018 Allan Leal
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

#include <catch.hpp>

// C++ includes
#include <iostream>
#include <iomanip>

// Eigen includes
#include <eigen3/Eigen/LU>
using namespace Eigen;

// Optima includes
#include <Optima/Core/IpSaddlePointMatrix.hpp>
#include <Optima/Core/IpSaddlePointSolver.hpp>
#include <Optima/Core/SaddlePointOptions.hpp>
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Math/Matrix.hpp>
using namespace Optima;

#define PRINT_STATE                                                         \
{                                                                           \
    std::cout << std::setprecision(10); \
    VectorXd slu = M.fullPivLu().solve(r);                                    \
    std::cout << "M = \n" << M << std::endl;                                \
    std::cout << "r        = " << tr(r) << std::endl;                           \
    std::cout << "x        = " << tr(s.head(n)) << std::endl;                        \
    std::cout << "x(lu)    = " << tr(slu.head(n)) << std::endl;      \
    std::cout << "y        = " << tr(s.segment(n, m)) << std::endl;                        \
    std::cout << "y(lu)    = " << tr(slu.segment(n, m)) << std::endl;      \
    std::cout << "z        = " << tr(s.segment(n+m, n)) << std::endl;                        \
    std::cout << "z(lu)    = " << tr(slu.segment(n+m, n)) << std::endl;      \
    std::cout << "w        = " << tr(s.tail(n)) << std::endl;                        \
    std::cout << "w(lu)    = " << tr(slu.tail(n)) << std::endl;      \
    std::cout << "res      = " << tr(M*s - r) << std::endl;                         \
    std::cout << "res(lu)  = " << tr(M*slu - r) << std::endl;                         \
}                                                                           \

//#undef PRINT_STATE
//#define PRINT_STATE

TEST_CASE("Testing IpSaddlePointSolver")
{
//    std::srand(std::time(0));
//    Index n = 60;
//    Index m = 20;
//    Index n = 6;
//    Index m = 3;
    Index n = 2;
    Index m = 1;
    Index t = 3*n + m;
    Index nx = n;
    Index nf = 0;
//
//    MatrixXd A = random(m, n);
//    MatrixXd H = random(n, n);
//    VectorXd Z = random(n);
//    VectorXd W = random(n);
//    VectorXd L = random(n);
//    VectorXd U = random(n);

    MatrixXd A = ones(m, n);
    MatrixXd H = ones(n, n);
    VectorXd Z = ones(n);
    VectorXd W = ones(n);
    VectorXd L = ones(n);
    VectorXd U = ones(n);

    const VectorXd expected = linspace(t, 1, t);

    SaddlePointOptions options;

    MatrixXd M = zeros(t, t);
    VectorXd r = zeros(t);

    auto check = [&]()
    {
        // The left-hand side coefficient matrix
        IpSaddlePointMatrix lhs(H, A, Z, W, L, U, nx, nf);

        // The dense matrix assembled from lhs
        const MatrixXd M = lhs.matrix();

        // The right-hand side vector
        const VectorXd r = M * expected;
        IpSaddlePointVector rhs(r, n, m);

        // The solution vector
        VectorXd s = zeros(t);
        IpSaddlePointSolution sol(s, n, m);

        // Solve the interior-poin saddle point problem
        IpSaddlePointSolver solver;
        solver.setOptions(options);
        solver.initialize(A);
        solver.decompose(lhs);
        solver.solve(rhs, sol);

        PRINT_STATE;

        // Check the residual of the equation Ms = r
        REQUIRE(norm(M*s - r)/norm(r) == Approx(0.0));
    };

//    SECTION("When all variables are free.")
//    {
//        check();
//    }
//
//    SECTION("When m variables are fixed.")
//    {
//        nx = n - m;
//        nf = m;
//
//        check();
//    }

    SECTION("When some entries in L are very small.")
    {
        L.head(1).fill(1e-16); // this works
//        L.tail(1).fill(1e-16); // this does not work

        check();
    }
}
