// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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

#include <Optima/Optima.hpp>
using namespace Optima;

int main(int argc, char **argv)
{
	OptimumProblem problem;
	problem.n = 2;
	problem.objective = [](const VectorXd& x, ObjectiveState& f)
	{
		VectorXd xt = x - ones(2);
		if(f.requires.val) f.val = 0.5 * dot(xt, xt);
		if(f.requires.grad) f.grad = xt;
		if(f.requires.hessian)
		{
			f.hessian.mode = Hessian::Diagonal;
			f.hessian.diagonal = ones(x.rows());
		}
	};
	problem.A.resize(1, 2);
	problem.A << 1, -1;

	problem.a.resize(1);
	problem.a << 0;

	OptimumState state;
	state.x.resize(2);
	state.x << 6, 3;

	OptimumOptions options;
	options.output.active = true;

	OptimumSolver solver;

	solver.solve(problem, state, options);

	std::cout << state.x << std::endl;
}



