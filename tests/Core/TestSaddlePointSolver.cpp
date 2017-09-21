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

// Eigen includes
#include <Eigenx/LU.hpp>
using namespace Eigen;

// Optima includes
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointOptions.hpp>
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Core/SaddlePointSolver.hpp>
#include <Optima/Math/Matrix.hpp>
using namespace Optima;

void testSaddlePointSolver(SaddlePointMatrix lhs, SaddlePointOptions options)
{
    Index m = lhs.A().rows();
    Index n = lhs.A().cols();
    Index t = m + n;

    VectorXd expected = linspace(t, 1, t);

    VectorXd r = lhs * expected;
    VectorXd s(t);

    SaddlePointVector rhs(r, n, m);
    SaddlePointSolution sol(s, n, m);

    SaddlePointSolver solver;
    solver.setOptions(options);
    solver.canonicalize(lhs.A());
    solver.decompose(lhs);
    solver.solve(lhs, rhs, sol);

    double error = (lhs.matrix() * s - r).norm()/r.norm();
    CHECK(approx(error) == 0.0);

//    SUBCASE("When the order of the variables change...")
    {
        VectorXi ordering(n);
        ordering.head(m).setLinSpaced(m, n - 1, n - m);
        ordering.tail(n - m).setLinSpaced(n - m, 0, n - m);
//        ordering.setLinSpaced(n, 0, n);
        const auto Q = ordering.asPermutation();
        MatrixXd A = lhs.A() * Q;
        MatrixXd H = Q.transpose() * lhs.H() * Q;
        VectorXi ifixed = rows(ordering, lhs.fixed());
        r.head(n) = Q.transpose() * r.head(n);
        SaddlePointMatrix lhsnew(H, A, lhs.G(), ifixed);
        solver.update(ordering); // no need to perform canonicalization again
        solver.decompose(lhsnew);
        solver.solve(lhsnew, rhs, sol);
        VectorXd s1 = lhsnew.matrix().fullPivLu().solve(r);
//        std::cout << "lhs = \n" << lhs.matrix() << std::endl;
//        std::cout << "newlhs = \n" << lhsnew.matrix() << std::endl;
        std::cout << "s1 = " << tr(s1) << std::endl;
        std::cout << "s2 = " << tr(s) << std::endl;
        error = (lhsnew.matrix() * s1 - r).norm()/r.norm();
        CHECK(approx(error) == 0.0);
    }

//    SUBCASE("When matrix G is empty and then equivalent to zero...")
//    {
//        MatrixXd G;
//        SaddlePointMatrix lhsnew(lhs.H(), lhs.A(), G, lhs.fixed());
//        solver.decompose(lhsnew);
//        solver.solve(lhsnew, rhs, sol);
//        double error = (lhsnew.matrix() * s - r).norm()/r.norm();
//        CHECK(approx(error) == 0.0);
//    }
}

#define TEST_SADDLE_POINT_SOLVER()                                             \
{                                                                              \
    SaddlePointMatrix lhs(H, A, G, ifixed);\
\
    VectorXd r = lhs * expected;\
    VectorXd s(t);\
\
    SaddlePointVector rhs(r, n, m);\
    SaddlePointSolution sol(s, n, m);\
\
    SUBCASE("When using FullPivLU")                                            \
    {                                                                          \
        options.method = SaddlePointMethod::FullPivLU;                         \
        SaddlePointSolver solver; solver.setOptions(options);                                            \
        solver.canonicalize(lhs.A());                                          \
        solver.decompose(lhs);                                                 \
        solver.solve(lhs, rhs, sol);                                           \
        double error = (lhs.matrix() * s - r).norm()/r.norm();                 \
        CHECK(approx(error) == 0.0);                                           \
    }                                                                          \
                                                                               \
    SUBCASE("When using PartialPivLU")                                         \
    {                                                                          \
        options.method = SaddlePointMethod::PartialPivLU;                      \
        SaddlePointSolver solver; solver.setOptions(options);                                            \
        solver.canonicalize(lhs.A());                                          \
        solver.decompose(lhs);                                                 \
        solver.solve(lhs, rhs, sol);                                           \
        double error = (lhs.matrix() * s - r).norm()/r.norm();                 \
        CHECK(approx(error) == 0.0);                                           \
    }                                                                          \
                                                                               \
    SUBCASE("When using Nullspace")                                            \
    {                                                                          \
        options.method = SaddlePointMethod::Nullspace;                         \
        SaddlePointSolver solver; solver.setOptions(options);                                            \
        solver.canonicalize(lhs.A());                                          \
        solver.decompose(lhs);                                                 \
        solver.solve(lhs, rhs, sol);                                           \
        double error = (lhs.matrix() * s - r).norm()/r.norm();                 \
        CHECK(approx(error) == 0.0);                                           \
    }                                                                          \
                                                                               \
    SUBCASE("When using RangespaceDiagonal")                                   \
    {                                                                          \
        MatrixXd H = diag(lhs.H().diagonal());                                 \
        SaddlePointMatrix lhsdiag(H, lhs.A(), lhs.G(), lhs.fixed());           \
        options.method = SaddlePointMethod::RangespaceDiagonal;                \
        SaddlePointSolver solver; solver.setOptions(options);                                            \
        solver.canonicalize(lhs.A());                                          \
        solver.decompose(lhsdiag);                                             \
        solver.solve(lhsdiag, rhs, sol);                                       \
        double error = (lhsdiag.matrix() * s - r).norm()/r.norm();             \
        CHECK(approx(error) == 0.0);                                           \
    }                                                                          \
}                                                                              \

TEST_CASE("Testing SaddlePointSolver with other methods.")
{
    const Index m = 3;
    const Index n = 10;
    const Index t = m + n;
//    Index m = 10;
//    Index n = 60;

    MatrixXd A = random(m, n);
    MatrixXd H = random(n, n);
    MatrixXd G = random(m, m);
//    MatrixXd G = diag(constants(m, 1e-3));
//    MatrixXd G;
//    MatrixXd G = zeros(m, m);
    VectorXi ifixed;

    SaddlePointOptions options;

    VectorXd expected = linspace(t, 1, t);

    SUBCASE("When there are no fixed variables")
    {
        TEST_SADDLE_POINT_SOLVER();
    }

    SUBCASE("When there are fixed variables")
    {
        ifixed.setLinSpaced(6, 0, 5);

        TEST_SADDLE_POINT_SOLVER();
    }

    SUBCASE("When there are many fixed variables enough to degenerate the problem")
    {
        // Modify matrix A so that its first row has zeros for all entries corresponding to free variables.
        // Set the fixed variables as the variables that have such entries positive.
        // This results in a submatrix for the free variables with the first row zero.
        ifixed = linspace<int>(n/2);
        A.row(0).rightCols(n/2).fill(0.0);

        TEST_SADDLE_POINT_SOLVER();
    }

    SUBCASE("When there are linearly dependent rows and many fixed variables enough to degenerate the problem")
    {
        // Modify matrix A so that its first row has zeros for all entries corresponding to free variables.
        // Set the fixed variables as the variables that have such entries positive.
        // This results in a submatrix for the free variables with the first row zero.
        ifixed = linspace<int>(n/2);
        A.row(0).rightCols(n/2).fill(0.0);

        // Set the 3rd row of A as the 2nd row to create the linear dependency
        A.row(2) = A.row(1);

        TEST_SADDLE_POINT_SOLVER();

//        SUBCASE("When A has large entries")
//        {
//            A *= 1e6;
//            TEST_SADDLE_POINT_SOLVER(options);
//        }
//
//        SUBCASE("When A has rational entries")
//        {
//            MatrixXi Anum = random<int>(m, n);
//            MatrixXi Aden = random<int>(m, n);
//            A = Anum.cast<double>()/Aden.cast<double>();
//
//            options.rationalize = true;
//            options.maxdenominator = Aden.maxCoeff() * 10;
//            TEST_SADDLE_POINT_SOLVER(options);
//        }
    }

    SUBCASE("When the system corresponds to one from a linear programming problem")
    {
        SUBCASE("When all variables are stable...")
        {
            H.fill(0.0);
            H.diagonal().fill(1e-13);
            TEST_SADDLE_POINT_SOLVER();
        }

        SUBCASE("When all variables are unstable...")
        {
            H.fill(0.0);
            H.diagonal().fill(1e+13);
            TEST_SADDLE_POINT_SOLVER();
        }

        SUBCASE("When `m` variables are stable...")
        {
            H.fill(0.0);
            H.diagonal().fill(1e+13);
            H.diagonal().head(m).fill(1e-13);
            TEST_SADDLE_POINT_SOLVER();
        }

        SUBCASE("When `m - 1` variables are stable...")
        {
            H.fill(0.0);
            H.diagonal().fill(1e+13);
            H.diagonal().head(m - 1).fill(1e-13);
            TEST_SADDLE_POINT_SOLVER();
        }
    }
}

