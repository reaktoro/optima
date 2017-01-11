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
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Core/SaddlePointSolver.hpp>
using namespace Optima;

TEST_CASE("Testing SaddlePointSolver with RangespaceDiagonal method.")
{
    Index m = 10;
    Index n = 60;
    Index t = m + n;

    VectorXd expected = linspace(t, 1, t);

    MatrixXd A = random(m, n);
    MatrixXd H = diag(random(n));

    SaddlePointMatrix lhs(H, A);

    VectorXd r = lhs * expected;
    VectorXd s(t);

    SaddlePointVector rhs(r, n, m);
    SaddlePointSolution sol(s, n, m);

    SaddlePointSolver solver;
    solver.setMethodRangespaceDiagonal();
    solver.canonicalize(lhs);
    solver.decompose(lhs);
    solver.solve(rhs, sol);

    CHECK_EQ(solver.method(), SaddlePointMethod::RangespaceDiagonal);
    CHECK(s.isApprox(expected));
}

TEST_CASE("Testing SaddlePointSolver with Nullspace method.")
{
    Index m = 10;
    Index n = 60;
    Index t = m + n;

    VectorXd expected = linspace(t, 1, t);

    MatrixXd A = random(m, n);
    MatrixXd H = random(n, n);

    SaddlePointMatrix lhs(H, A);

    VectorXd r = lhs * expected;
    VectorXd s(t);

    SaddlePointVector rhs(r, n, m);
    SaddlePointSolution sol(s, n, m);

    SUBCASE("When method is SaddlePointMethod::Nullspace")
    {
        SaddlePointSolver solver;
        solver.setMethodNullspace();
        solver.canonicalize(lhs);
        solver.decompose(lhs);
        solver.solve(rhs, sol);
        CHECK_EQ(solver.method(), SaddlePointMethod::Nullspace);
        CHECK(s.isApprox(expected));
    }

    SUBCASE("When method is SaddlePointMethod::PartialPivLU")
    {
        SaddlePointSolver solver;
        solver.setMethodPartialPivLU();
        solver.canonicalize(lhs);
        solver.decompose(lhs);
        solver.solve(rhs, sol);
        CHECK_EQ(solver.method(), SaddlePointMethod::PartialPivLU);
        CHECK(s.isApprox(expected));
    }

    SUBCASE("When method is SaddlePointMethod::FullPivLU")
    {
        SaddlePointSolver solver;
        solver.setMethodFullPivLU();
        solver.canonicalize(lhs);
        solver.decompose(lhs);
        solver.solve(rhs, sol);
        CHECK_EQ(solver.method(), SaddlePointMethod::FullPivLU);
        CHECK(s.isApprox(expected));
    }
}

