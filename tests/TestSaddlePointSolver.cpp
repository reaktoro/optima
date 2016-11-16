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

// Optima includes
#include <Optima/Optima.hpp>
using namespace Optima;

namespace Optima {


auto operator<<(std::ostream& out, const SaddlePointProblemCanonical& problem) -> std::ostream&
{
    // Auxiliary alias to problem data members
    const auto& Gb   = problem.Gb;
    const auto& Gs   = problem.Gs;
    const auto& Gu   = problem.Gu;
    const auto& Bb   = problem.Bb;
    const auto& Bs   = problem.Bs;
    const auto& Bu   = problem.Bu;
    const auto& Eb   = problem.Eb;
    const auto& Es   = problem.Es;
    const auto& Eu   = problem.Eu;
//    const auto& ab   = problem.ab;
//    const auto& as   = problem.as;
//    const auto& au   = problem.au;
//    const auto& b    = problem.b;
//    const auto& cb   = problem.cb;
//    const auto& cs   = problem.cs;
//    const auto& cu   = problem.cu;

    const auto nb = Gb.rows();
    const auto ns = Gs.rows();
    const auto nu = Gu.rows();
    const auto n = nb + ns + nu;
    const auto m = nb;
    const auto t = 2*n + m;

    Vector G(n);
    if(nb) G.topRows(nb) = Gb;
    if(ns) G.middleRows(nb, ns) = Gs;
    if(nu) G.bottomRows(nu) = Gu;

    Vector E(n);
    if(nb) E.topRows(nb) = Eb;
    if(ns) E.middleRows(nb, ns) = Es;
    if(nu) E.bottomRows(nu) = Eu;

    Matrix B(m, n);
    if(nb) B.leftCols(nb) = diag(Bb);
    if(ns) B.middleCols(nb, ns) = Bs;
    if(nu) B.rightCols(nu) = Bu;

    Matrix A(t, t);
    A.topLeftCorner(n, n) = diag(G);
    A.topRightCorner(n, n) = diag(E);
    A.middleRows(n, m).leftCols(n) = B;
    A.middleCols(n, m).topRows(n) = tr(B);
    A.bottomLeftCorner(n, n) = diag(E);
    A.bottomRightCorner(n, n) = diag(E);

    out << "[";
    for(auto i = 0; i < t; ++i)
    {
        out << "[";
        for(auto j = 0; j < t; ++j)
            out << (j > 0 ? ", " : "") << A(i, j);
        out << "];\n";
    }
    out << "]";

//    out << A;

    return out;
}

auto testSaddlePointSolver1() -> void
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

    std::cout << solution.xb << std::endl;
    std::cout << solution.xs << std::endl;
    std::cout << solution.xu << std::endl;
    std::cout << solution.y  << std::endl;
    std::cout << solution.zb << std::endl;
    std::cout << solution.zs << std::endl;
    std::cout << solution.zu << std::endl;
}

auto testSaddlePointSolver2() -> void
{
    SaddlePointProblemCanonical problem;
    SaddlePointSolutionCanonical solution;

    problem.Gb = {5};
    problem.Gs = {5};
    problem.Gu = {};

    problem.Bb = {2};
    problem.Bs = {2};
    problem.Bu = {};

    problem.Eb = {1};
    problem.Es = {1};
    problem.Eu = {};

    problem.ab = {8};
    problem.as = {8};
    problem.au = {};
    problem.b  = {4};
    problem.cb = {2};
    problem.cs = {2};
    problem.cu = {};

    std::cout << problem << std::endl;
    /**/
    solver(problem, solution);

    std::cout << solution.xb << std::endl;
    std::cout << solution.xs << std::endl;
    std::cout << solution.xu << std::endl;
    std::cout << solution.y  << std::endl;
    std::cout << solution.zb << std::endl;
    std::cout << solution.zs << std::endl;
    std::cout << solution.zu << std::endl;
    //*/
}

auto testSaddlePointSolver3() -> void
{
    SaddlePointProblemCanonical problem;
    SaddlePointSolutionCanonical solution;

    problem.Gb = {5};
    problem.Gs = {3};
    problem.Gu = {1};

    problem.Bb = {2};
    problem.Bs = {2};
    problem.Bu = {2};

    problem.Eb = {1};
    problem.Es = {1};
    problem.Eu = {6};

    problem.ab = {8};
    problem.as = {6};
    problem.au = {9};
    problem.b  = {6};
    problem.cb = {2};
    problem.cs = {2};
    problem.cu = {12};

    std::cout << problem << std::endl;
    /**/
    solver(problem, solution);

    std::cout << solution.xb << std::endl;
    std::cout << solution.xs << std::endl;
    std::cout << solution.xu << std::endl;
    std::cout << solution.y  << std::endl;
    std::cout << solution.zb << std::endl;
    std::cout << solution.zs << std::endl;
    std::cout << solution.zu << std::endl;
    //*/
}

auto testSaddlePointSolver4() -> void
{
    SaddlePointProblemCanonical problem;
    SaddlePointSolutionCanonical solution;

    problem.Gb = {8, 7, 6};
    problem.Gs = {6, 5};
    problem.Gu = {4};

    problem.Bb = {9, 8, 7};
    problem.Bs = {{1, 2}, {2, 3}, {3, 4}};
    problem.Bu = {5, 5, 5};

    problem.Eb = {1, 2, 3};
    problem.Es = {1, 3};
    problem.Eu = {9};

    problem.ab = {18, 17, 16};
    problem.as = {13, 17};
    problem.au = {28};
    problem.b  = {17, 18, 19};
    problem.cb = {2, 4, 6};
    problem.cs = {2, 6};
    problem.cu = {18};

//    std::cout << problem << std::endl;
    /**/
    solver(problem, solution);

    std::cout << solution.xb << std::endl;
    std::cout << solution.xs << std::endl;
    std::cout << solution.xu << std::endl;
    std::cout << solution.y  << std::endl;
    std::cout << solution.zb << std::endl;
    std::cout << solution.zs << std::endl;
    std::cout << solution.zu << std::endl;
    //*/
}

} // namespace Optima

int main(int argc, char **argv)
{
    testSaddlePointSolver4();
}



