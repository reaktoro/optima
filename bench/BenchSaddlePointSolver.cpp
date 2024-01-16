// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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

// C++ includes
#include <iostream>

// Optima includes
#include <Optima/Matrix.hpp>
#include <Optima/SaddlePointMatrix.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/SaddlePointSolver.hpp>
using namespace Optima;

// Eigen includes
#include <eigenx/Eigen/LU>
using Eigen::PartialPivLU;
using Eigen::FullPivLU;

Index samples = 10;

void benchMethodRangespace()
{
    Index m = 10;
    Index n = 60;
    Index t = m + n;

    Vector expected = linspace(t, 1, t);

    Matrix A = random(m, n);
    Matrix H = diag(random(n));
    Indices ifixed;

    SaddlePointMatrix lhs(H, A, ifixed);

    Matrix M = lhs.matrix();
    Vector r = M * expected;
    Vector s(t);

    SaddlePointVector rhs(r, n, m);
    SaddlePointSolution sol(s, n, m);
    SaddlePointOptions options;

    Result res1, res2, res3, res;

    for(Index i = 0; i < samples; ++i)
    {
        options.method = SaddlePointMethod::Rangespace;

        SaddlePointSolver solver;
        solver.setOptions(options);
        res1 += solver.canonicalize(A);
        res2 += solver.decompose(lhs);
        res3 += solver.solve(rhs, sol);

        res += res1 + res2 + res3;
    }

    double timesps1 = res1.time()/samples;
    double timesps2 = res2.time()/samples;
    double timesps3 = res3.time()/samples;
    double timesps  = timesps1 + timesps2 + timesps3;

    double time_partiallu1 = 0.0, time_partiallu2 = 0.0, time_partiallu = 0.0;
    double time_fulllu1 = 0.0, time_fulllu2 = 0.0, time_fulllu = 0.0;

    Vector s_partiallu(n + m);
    Vector s_fulllu(n + m);

    Time begin;

    for(Index i = 0; i < samples; ++i)
    {
        PartialPivLU<Matrix> partiallu(M);
        FullPivLU<Matrix> fulllu(M);

        begin = timenow();
        partiallu.compute(M);
        time_partiallu1 += elapsed(begin);

        begin = timenow();
        fulllu.compute(M);
        time_fulllu1 += elapsed(begin);

        begin = timenow();
        s_partiallu.noalias() = partiallu.solve(r);
        time_partiallu2 += elapsed(begin);

        begin = timenow();
        s_fulllu.noalias() = fulllu.solve(r);
        time_fulllu2 += elapsed(begin);
    }

    time_partiallu1 /= samples; time_partiallu2 /= samples; time_partiallu = time_partiallu1 + time_partiallu2;
    time_fulllu1 /= samples; time_fulllu2 /= samples; time_fulllu = time_fulllu1 + time_fulllu2;

    std::cout << std::endl;
    std::cout << "=============================================================" << std::endl;
    std::cout << "Saddle Point Solver Analysis: Rangespace Diagonal Method" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Error(SaddlePointSolver):              " << norminf(s - expected) << std::endl;
    std::cout << "Error(PartialPivLU):                   " << norminf(s_partiallu - expected) << std::endl;
    std::cout << "Error(FullPivLU):                      " << norminf(s_fulllu - expected) << std::endl;
    std::cout << std::endl;
    std::cout << "Time(SaddlePointSolver::canonicalize): " << timesps1 << std::endl;
    std::cout << "Time(SaddlePointSolver::decompose):    " << timesps2 << std::endl;
    std::cout << "Time(SaddlePointSolver::solve):        " << timesps3 << std::endl;
    std::cout << "Time(SaddlePointSolver::all):          " << timesps  << std::endl;
    std::cout << std::endl;
    std::cout << "Time(PartialPivLU::decompose):         " << time_partiallu1 << std::endl;
    std::cout << "Time(PartialPivLU::solve):             " << time_partiallu2 << std::endl;
    std::cout << std::endl;
    std::cout << "Time(FullPivLU::decompose):            " << time_fulllu1 << std::endl;
    std::cout << "Time(FullPivLU::solve):                " << time_fulllu2 << std::endl;
    std::cout << std::endl;
    std::cout << "Speedup(PartialPivLU): " << std::endl;
    std::cout << "Speedup(canonicalize+decompose):       " << time_partiallu1/(timesps1 + timesps2) << std::endl;
    std::cout << "Speedup(decompose):                    " << time_partiallu1/timesps2 << std::endl;
    std::cout << "Speedup(solve):                        " << time_partiallu2/timesps3 << std::endl;
    std::cout << "Speedup(decompose+solve):              " << time_partiallu/(timesps2 + timesps3) << std::endl;
    std::cout << std::endl;
    std::cout << "Speedup(FullPivLU): " << std::endl;
    std::cout << "Speedup(canonicalize+decompose):       " << time_fulllu1/(timesps1 + timesps2) << std::endl;
    std::cout << "Speedup(decompose):                    " << time_fulllu1/timesps2 << std::endl;
    std::cout << "Speedup(solve):                        " << time_fulllu2/timesps3 << std::endl;
    std::cout << "Speedup(decompose+solve):              " << time_fulllu/(timesps2 + timesps3) << std::endl;
    std::cout << "=============================================================" << std::endl;
}

void benchMethodNullspace()
{
    Index m = 10;
    Index n = 60;
    Index t = m + n;

    Vector expected = linspace(t, 1, t);

    Matrix A = random(m, n);
    Matrix H = random(n, n);
    H.diagonal().head(m)   *= 1e-2;
    H.diagonal().tail(n-m) *= 1e+5;
    Indices ifixed;

    SaddlePointMatrix lhs(H, A, ifixed);

    Matrix M = lhs.matrix();
    Vector r = M * expected;
    Vector s(t);

    SaddlePointVector rhs(r, n, m);
    SaddlePointSolution sol(s, n, m);
    SaddlePointOptions options;

    Result res1, res2, res3, res;

    for(Index i = 0; i < samples; ++i)
    {
        options.method = SaddlePointMethod::Nullspace;

        SaddlePointSolver solver;
        solver.setOptions(options);
        res1 += solver.canonicalize(A);
        res2 += solver.decompose(lhs);
        res3 += solver.solve(rhs, sol);

        res += res1 + res2 + res3;
    }

    double timesps1 = res1.time()/samples;
    double timesps2 = res2.time()/samples;
    double timesps3 = res3.time()/samples;
    double timesps  = timesps1 + timesps2 + timesps3;

    double time_partiallu1 = 0.0, time_partiallu2 = 0.0, time_partiallu = 0.0;
    double time_fulllu1 = 0.0, time_fulllu2 = 0.0, time_fulllu = 0.0;

    Vector s_partiallu(n + m);
    Vector s_fulllu(n + m);

    Time begin;

    for(Index i = 0; i < samples; ++i)
    {
        PartialPivLU<Matrix> partiallu(M);
        FullPivLU<Matrix> fulllu(M);

        begin = timenow();
        partiallu.compute(M);
        time_partiallu1 += elapsed(begin);

        begin = timenow();
        fulllu.compute(M);
        time_fulllu1 += elapsed(begin);

        begin = timenow();
        s_partiallu.noalias() = partiallu.solve(r);
        time_partiallu2 += elapsed(begin);

        begin = timenow();
        s_fulllu.noalias() = fulllu.solve(r);
        time_fulllu2 += elapsed(begin);
    }

    time_partiallu1 /= samples; time_partiallu2 /= samples; time_partiallu = time_partiallu1 + time_partiallu2;
    time_fulllu1 /= samples; time_fulllu2 /= samples; time_fulllu = time_fulllu1 + time_fulllu2;

    std::cout << std::endl;
    std::cout << "=============================================================" << std::endl;
    std::cout << "Saddle Point Solver Analysis: Nullspace Method" << std::endl;
    std::cout << "-------------------------------------------------------------" << std::endl;
    std::cout << "Error(SaddlePointSolver):              " << norminf(s - expected) << std::endl;
    std::cout << "Error(PartialPivLU):                   " << norminf(s_partiallu - expected) << std::endl;
    std::cout << "Error(FullPivLU):                      " << norminf(s_fulllu - expected) << std::endl;
    std::cout << std::endl;
    std::cout << "Time(SaddlePointSolver::canonicalize): " << timesps1 << std::endl;
    std::cout << "Time(SaddlePointSolver::decompose):    " << timesps2 << std::endl;
    std::cout << "Time(SaddlePointSolver::solve):        " << timesps3 << std::endl;
    std::cout << "Time(SaddlePointSolver::all):          " << timesps  << std::endl;
    std::cout << std::endl;
    std::cout << "Time(PartialPivLU::decompose):         " << time_partiallu1 << std::endl;
    std::cout << "Time(PartialPivLU::solve):             " << time_partiallu2 << std::endl;
    std::cout << std::endl;
    std::cout << "Time(FullPivLU::decompose):            " << time_fulllu1 << std::endl;
    std::cout << "Time(FullPivLU::solve):                " << time_fulllu2 << std::endl;
    std::cout << std::endl;
    std::cout << "Speedup(PartialPivLU): " << std::endl;
    std::cout << "Speedup(canonicalize+decompose):       " << time_partiallu1/(timesps1 + timesps2) << std::endl;
    std::cout << "Speedup(decompose):                    " << time_partiallu1/timesps2 << std::endl;
    std::cout << "Speedup(solve):                        " << time_partiallu2/timesps3 << std::endl;
    std::cout << "Speedup(decompose+solve):              " << time_partiallu/(timesps2 + timesps3) << std::endl;
    std::cout << std::endl;
    std::cout << "Speedup(FullPivLU): " << std::endl;
    std::cout << "Speedup(canonicalize+decompose):       " << time_fulllu1/(timesps1 + timesps2) << std::endl;
    std::cout << "Speedup(decompose):                    " << time_fulllu1/timesps2 << std::endl;
    std::cout << "Speedup(solve):                        " << time_fulllu2/timesps3 << std::endl;
    std::cout << "Speedup(decompose+solve):              " << time_fulllu/(timesps2 + timesps3) << std::endl;
    std::cout << "=============================================================" << std::endl;
}

int main(int argc, char **argv)
{
    benchMethodRangespace();
    benchMethodNullspace();
}


