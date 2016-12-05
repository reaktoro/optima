// Optima is a C++ library for numerical sol of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2016 Allan Leal
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
using namespace Optima;

namespace util {

/// Return a random SaddlePointProblem instance.
/// @param m The number of rows in matrix A
/// @param n The number of columns in matrix A
auto saddlePointProblem(Index m, Index n) -> SaddlePointProblem
{
    SaddlePointProblem problem;
    problem.lhs.H = random(n);
    problem.lhs.A = random(m, n);
    problem.lhs.X = random(n);
    problem.lhs.Z = random(n);

    Vector rhs = problem.lhs * ones(problem.lhs.rows());

    problem.rhs.a = rhs.head(n);
    problem.rhs.b = rhs.segment(n, m);
    problem.rhs.c = rhs.tail(n);

    REQUIRE(problem.lhs.valid());
    REQUIRE(problem.rhs.valid());

    return problem;
}

/// Return a random SaddlePointProblemCanonical instance.
/// @param nb The number of columns in matrix Bb
/// @param ns The number of columns in matrix Bs
/// @param nu The number of columns in matrix Bu
/// @param p  A number than if equal to zero means E is empty, otherwise E is a random diagonal matrix.
auto saddlePointProblemCanonical(Index nb, Index ns, Index nu, Index p) -> SaddlePointProblemCanonical
{
	Index n = nb + ns + nu;
	Index m = nb;

	SaddlePointProblemCanonical problem;
    problem.lhs.Gb = random(nb);
    problem.lhs.Gs = random(ns);
    problem.lhs.Gu = random(nu);
    problem.lhs.Bb = random(nb);
    problem.lhs.Bs = random(nb, ns);
    problem.lhs.Bu = random(nb, nu);
    problem.lhs.Eb = random(p ? nb : 0);
    problem.lhs.Es = random(p ? ns : 0);
    problem.lhs.Eu = random(p ? nu : 0);

    Vector rhs = problem.lhs * ones(problem.lhs.rows());

    problem.rhs.rb = rhs.topRows(n).topRows(nb);
    problem.rhs.rs = rhs.topRows(n).middleRows(nb, ns);
    problem.rhs.ru = rhs.topRows(n).bottomRows(nu);
    problem.rhs.s  = rhs.middleRows(n, m);
    problem.rhs.tb = rhs.bottomRows(n).topRows(nb);
    problem.rhs.ts = rhs.bottomRows(n).middleRows(nb, ns);
    problem.rhs.tu = rhs.bottomRows(n).bottomRows(nu);

    REQUIRE(problem.lhs.valid());
	REQUIRE(problem.rhs.valid());

	return problem;
}

} // namespace util

TEST_CASE("Testing the solution of a general saddle point problem: Case 1")
{
    Index m = 3;
    Index n = 6;

    SaddlePointVector sol;
    SaddlePointProblem problem = util::saddlePointProblem(m, n);

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}

TEST_CASE("Testing the solution of a canonical saddle point problem: Case 1")
{
    SaddlePointProblemCanonical problem;
    SaddlePointVectorCanonical sol;

    problem.lhs.Gb = {9, 8, 7};
    problem.lhs.Gs = {};
    problem.lhs.Gu = {};
    problem.lhs.Bb = {1, 1, 1};
    problem.lhs.Bs = {};
    problem.lhs.Bu = {};
    problem.lhs.Eb = {1, 1, 1};
    problem.lhs.Es = {};
    problem.lhs.Eu = {};

    problem.rhs.rb = {11, 10, 9};
    problem.rhs.rs = {};
    problem.rhs.ru = {};
    problem.rhs.s  = {1, 1, 1};
    problem.rhs.tb = {2, 2, 2};
    problem.rhs.ts = {};
    problem.rhs.tu = {};

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}

TEST_CASE("Testing the solution of a canonical saddle point problem: Case 2")
{
    Index np = 10;
    Index ns = 35;
    Index nu = 5;
    Index p  = 1;

    SaddlePointVectorCanonical sol;
    SaddlePointProblemCanonical problem =
		util::saddlePointProblemCanonical(np, ns, nu, p);

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}

TEST_CASE("Testing the solution of a canonical saddle point problem: Case 3")
{
    SaddlePointVectorCanonical sol;
    SaddlePointProblemCanonical problem;

    Index np = 10;
    Index ns = 35;
    Index nu = 0;
    Index p  = 0;

    problem = util::saddlePointProblemCanonical(np, ns, nu, p);

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}

TEST_CASE("Testing the solution of a canonical saddle point problem: Case 4")
{
    SaddlePointVectorCanonical sol;
    SaddlePointProblemCanonical problem;

    Index np = 10;
    Index ns = 0;
    Index nu = 0;
    Index p  = 0;

    problem = util::saddlePointProblemCanonical(np, ns, nu, p);

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}

TEST_CASE("Testing the solution of a canonical saddle point problem: Case 4")
{
    SaddlePointVectorCanonical sol;
    SaddlePointProblemCanonical problem;

    Index np = 10;
    Index ns = 0;
    Index nu = 0;
    Index p  = 1;

    problem = util::saddlePointProblemCanonical(np, ns, nu, p);

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}

TEST_CASE("Testing the solution of a canonical saddle point problem: Case 5")
{
    SaddlePointProblemCanonical problem;
    SaddlePointVectorCanonical sol;

    problem.lhs.Gb = {5};
    problem.lhs.Gs = {5};
    problem.lhs.Gu = {};
    problem.lhs.Bb = {2};
    problem.lhs.Bs = {2};
    problem.lhs.Bu = {};
    problem.lhs.Eb = {1};
    problem.lhs.Es = {1};
    problem.lhs.Eu = {};

    problem.rhs.rb = {8};
    problem.rhs.rs = {8};
    problem.rhs.ru = {};
    problem.rhs.s  = {4};
    problem.rhs.tb = {2};
    problem.rhs.ts = {2};
    problem.rhs.tu = {};

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}

TEST_CASE("Testing the solution of a canonical saddle point problem: Case 6")
{
    SaddlePointProblemCanonical problem;
    SaddlePointVectorCanonical sol;

    problem.lhs.Gb = {5};
    problem.lhs.Gs = {};
    problem.lhs.Gu = {1};
    problem.lhs.Bb = {2};
    problem.lhs.Bs = {};
    problem.lhs.Bu = {2};
    problem.lhs.Eb = {1};
    problem.lhs.Es = {};
    problem.lhs.Eu = {6};

    problem.rhs.rb = {8};
    problem.rhs.rs = {};
    problem.rhs.ru = {9};
    problem.rhs.s  = {4};
    problem.rhs.tb = {2};
    problem.rhs.ts = {};
    problem.rhs.tu = {12};

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}

TEST_CASE("Testing the solution of a canonical saddle point problem: Case 7")
{
    SaddlePointProblemCanonical problem;
    SaddlePointVectorCanonical sol;

    problem.lhs.Gb = {1, 2, 3};
    problem.lhs.Gs = {4, 5};
    problem.lhs.Gu = {6};
    problem.lhs.Bb = {9, 8, 7};
    problem.lhs.Bs = {{1, 2}, {2, 3}, {3, 4}};
    problem.lhs.Bu = {5, 6, 7};
    problem.lhs.Eb = {1, 1, 1};
    problem.lhs.Es = {1, 1};
    problem.lhs.Eu = {1};

    problem.rhs.rb = {11, 11, 11};
    problem.rhs.rs = {11, 15};
    problem.rhs.ru = {25};
    problem.rhs.s  = {17, 19, 21};
    problem.rhs.tb = {2, 2, 2};
    problem.rhs.ts = {2, 2};
    problem.rhs.tu = {2};

    SaddlePointSolver solver;
    solver.solve(problem, sol);

    CHECK(sol.isApprox(ones(sol.rows())));
}
