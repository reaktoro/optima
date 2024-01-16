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

#include <Optima/Optima.hpp>
using namespace Optima;

int main(int argc, char **argv)
{
    // Solve the following problem:
    // min( (x-1)**2 + (y-1)**2 ) subject to x = y and x,y >= 0

    Dims dims;
    dims.x = 2; // number of variables
    dims.be = 1; // number of linear equality constraints

    Problem problem(dims);
    problem.Aex = Matrix{{ {1.0, -1.0} }};
    problem.be = Vector{{ 0.0 }};
    problem.f = [](ObjectiveResultRef res, VectorView x, VectorView p, VectorView c, ObjectiveOptions opts)
    {
        res.f = (x[0] - 1)*(x[0] - 1) + (x[1] - 1)*(x[1] - 1);
        res.fx = 2.0 * (x - 1);
        res.fxx = 2.0 * identity(2, 2);
    };

    State state(dims);

    Options options;
    options.output.active = true;

    Solver solver;
    solver.setOptions(options);

    solver.solve(problem, state);

    std::cout << state.x << std::endl;
}



