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
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Core/SaddlePointSolver.hpp>
#include <Optima/Math/Matrix.hpp>
using namespace Optima;
using namespace Eigen;

void testSaddlePointSolver(SaddlePointMatrix lhs, SaddlePointMethod method)
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
    solver.setMethod(method);
    solver.canonicalize(lhs.A());
    solver.decompose(lhs);
    solver.solve(rhs, sol);

    double error = (lhs.matrix() * s - r).norm()/r.norm();

    CHECK(approx(error) == 0.0);
}

#define TEST_SADDLE_POINT_SOLVER                                                      \
{                                                                                     \
    SUBCASE("When using FullPivLU")                                                   \
    {                                                                                 \
        testSaddlePointSolver(lhs, SaddlePointMethod::FullPivLU);                     \
    }                                                                                 \
                                                                                      \
    SUBCASE("When using PartialPivLU")                                                \
    {                                                                                 \
        testSaddlePointSolver(lhs, SaddlePointMethod::PartialPivLU);                  \
    }                                                                                 \
                                                                                      \
    SUBCASE("When using Nullspace")                                                   \
    {                                                                                 \
        testSaddlePointSolver(lhs, SaddlePointMethod::Nullspace);                     \
    }                                                                                 \
                                                                                      \
    SUBCASE("When using RangespaceDiagonal")                                          \
    {                                                                                 \
        MatrixXd H = diag(random(lhs.H().rows()));                                    \
        SaddlePointMatrix lhsdiag(H, lhs.A(), lhs.fixed());                           \
        testSaddlePointSolver(lhsdiag, SaddlePointMethod::RangespaceDiagonal);        \
    }                                                                                 \
}                                                                                     \

TEST_CASE("Testing SaddlePointSolver with other methods.")
{
    Index m = 10;
    Index n = 60;

    MatrixXd A = random(m, n);
    MatrixXd H = random(n, n);

    SUBCASE("When there are no fixed variables")
    {
        SaddlePointMatrix lhs(H, A);

        TEST_SADDLE_POINT_SOLVER;
    }

    SUBCASE("When there are fixed variables")
    {
        Indices ifixed = {0, 10, 20, 30, 40, 50};

        SaddlePointMatrix lhs(H, A, ifixed);

        TEST_SADDLE_POINT_SOLVER;
    }

    SUBCASE("When there are many fixed variables enough to degenerate the problem")
    {
        // Modify matrix A so that its first row has zeros for all non-positive entries.
        // Set the fixed variables as the variables that have such entries positive.
        // This results in a submatrix for the free variables with the first row zero.
        Indices ifixed;
        for(Index i = 0; i < n; ++i)
            if(A(0, i) <= 0) A(0, i) = 0.0;
            else ifixed.push_back(i);

        SaddlePointMatrix lhs(H, A, ifixed);

        TEST_SADDLE_POINT_SOLVER;
    }

    SUBCASE("When there are linearly dependent rows and many fixed variables enough to degenerate the problem")
    {
        // Modify matrix A so that its first row has zeros for all non-positive entries.
        // Set the fixed variables as the variables that have such entries positive.
        // This results in a submatrix for the free variables with the first row zero.
        Indices ifixed;
        for(Index i = 0; i < n; ++i)
            if(A(0, i) <= 0) A(0, i) = 0.0;
            else ifixed.push_back(i);

        // Set the 3rd row of A as the 2nd row
        A.row(2) = A.row(1);

        SaddlePointMatrix lhs(H, A, ifixed);

        TEST_SADDLE_POINT_SOLVER;

        SUBCASE("When A has large entries")
        {
            A *= 1e6;
            TEST_SADDLE_POINT_SOLVER;
        }
    }
}

