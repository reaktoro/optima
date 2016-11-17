// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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

TEST_CASE("Testing SaddlePointSolver - Case 1")
{
    SaddlePointProblemCanonical problem;
    SaddlePointSolutionCanonical solution;

    problem.Gb = {9, 8, 7};
    problem.Gs = {};
    problem.Gu = {};

    problem.Bb = {1, 1, 1};
    problem.Bs = {};
    problem.Bu = {};

    problem.Eb = {1, 1, 1};
    problem.Es = {};
    problem.Eu = {};

    problem.ab = {11, 10, 9};
    problem.as = {};
    problem.au = {};
    problem.b  = {1, 1, 1};
    problem.cb = {2, 2, 2};
    problem.cs = {};
    problem.cu = {};

    solver(problem, solution);

    Vector expected = ones(6);

//    CHECK(expected.isApprox(solution));

//    std::cout << solution.xb << std::endl;
//    std::cout << solution.xs << std::endl;
//    std::cout << solution.xu << std::endl;
//    std::cout << solution.y  << std::endl;
//    std::cout << solution.zb << std::endl;
//    std::cout << solution.zs << std::endl;
//    std::cout << solution.zu << std::endl;
}

//auto testSaddlePointSolver2() -> bool
//{
//    SaddlePointProblemCanonical problem;
//    SaddlePointSolutionCanonical solution;
//
//    problem.Gb = {5};
//    problem.Gs = {5};
//    problem.Gu = {};
//
//    problem.Bb = {2};
//    problem.Bs = {2};
//    problem.Bu = {};
//
//    problem.Eb = {1};
//    problem.Es = {1};
//    problem.Eu = {};
//
//    problem.ab = {8};
//    problem.as = {8};
//    problem.au = {};
//    problem.b  = {4};
//    problem.cb = {2};
//    problem.cs = {2};
//    problem.cu = {};
//
//    solver(problem, solution);
//
////    std::cout << solution.xb << std::endl;
////    std::cout << solution.xs << std::endl;
////    std::cout << solution.xu << std::endl;
////    std::cout << solution.y  << std::endl;
////    std::cout << solution.zb << std::endl;
////    std::cout << solution.zs << std::endl;
////    std::cout << solution.zu << std::endl;
//
//    return true;
//}
//
//auto testSaddlePointSolver3() -> bool
//{
//    SaddlePointProblemCanonical problem;
//    SaddlePointSolutionCanonical solution;
//
//    problem.Gb = {5};
//    problem.Gs = {};
//    problem.Gu = {1};
//
//    problem.Bb = {2};
//    problem.Bs = {};
//    problem.Bu = {2};
//
//    problem.Eb = {1};
//    problem.Es = {};
//    problem.Eu = {6};
//
//    problem.ab = {8};
//    problem.as = {};
//    problem.au = {9};
//    problem.b  = {4};
//    problem.cb = {2};
//    problem.cs = {};
//    problem.cu = {12};
//
//    solver(problem, solution);
//
////    std::cout << solution.xb << std::endl;
////    std::cout << solution.xs << std::endl;
////    std::cout << solution.xu << std::endl;
////    std::cout << solution.y  << std::endl;
////    std::cout << solution.zb << std::endl;
////    std::cout << solution.zs << std::endl;
////    std::cout << solution.zu << std::endl;
//    //*/
//
//    return true;
//}
//
//auto testSaddlePointSolver4() -> bool
//{
//    SaddlePointProblemCanonical problem;
//    SaddlePointSolutionCanonical solution;
//
//    problem.Gb = {1, 2, 3};
//    problem.Gs = {4, 5};
//    problem.Gu = {6};
//
//    problem.Bb = {9, 8, 7};
//    problem.Bs = {{1, 2}, {2, 3}, {3, 4}};
//    problem.Bu = {5, 6, 7};
//
//    problem.Eb = {1, 1, 1};
//    problem.Es = {1, 1};
//    problem.Eu = {1};
//
//    problem.ab = {11, 11, 11};
//    problem.as = {11, 15};
//    problem.au = {25};
//    problem.b  = {17, 19, 21};
//    problem.cb = {2, 2, 2};
//    problem.cs = {2, 2};
//    problem.cu = {2};
//
////    problem.Gb = {8, 7, 6};
////    problem.Gs = {6, 5};
////    problem.Gu = {4};
////
////    problem.Bb = {9, 8, 7};
////    problem.Bs = {{1, 2}, {2, 3}, {3, 4}};
////    problem.Bu = {5, 5, 5};
////
////    problem.Eb = {1, 2, 3};
////    problem.Es = {1, 3};
////    problem.Eu = {9};
////
////    problem.ab = {18, 17, 16};
////    problem.as = {13, 17};
////    problem.au = {28};
////    problem.b  = {17, 18, 19};
////    problem.cb = {2, 4, 6};
////    problem.cs = {2, 6};
////    problem.cu = {18};
//
//    std::cout << problem << std::endl;
//    solver(problem, solution);
//
////    std::cout << solution.xb << std::endl;
////    std::cout << solution.xs << std::endl;
////    std::cout << solution.xu << std::endl;
////    std::cout << solution.y  << std::endl;
////    std::cout << solution.zb << std::endl;
////    std::cout << solution.zs << std::endl;
////    std::cout << solution.zu << std::endl;
//    return true;
//}
//
//TEST_CASE("Testing SaddlePointSolver...")
//{
//    testSaddlePointSolver1();
//    testSaddlePointSolver2();
//    testSaddlePointSolver3();
//    testSaddlePointSolver4();
//}



