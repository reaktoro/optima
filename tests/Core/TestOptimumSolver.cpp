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

#include <catch.hpp>

// Optima includes
//#include <Optima/Core/OptimumOptions.hpp>
//#include <Optima/Core/OptimumParams.hpp>
//#include <Optima/Core/OptimumProblem.hpp>
//#include <Optima/Core/OptimumResult.hpp>
//#include <Optima/Core/OptimumSolver.hpp>
//#include <Optima/Core/OptimumState.hpp>
//#include <Optima/Core/OptimumStructure.hpp>
//#include <Optima/Math/Matrix.hpp>
//using namespace Eigen;
//using namespace Optima;

TEST_CASE("Testing OptimumSolver")
{
//    const Index n = 10;
//    const Index m = 3;
//    const VectorXd c = abs(random(n));
//
//    OptimumStructure structure;
//    structure.n = n;
//    structure.A = random(m, n);
//    structure.objective = [&](VectorXdConstRef x, ObjectiveState& f)
//    {
//        f.val = dot(c, x);
//        f.grad = c;
//        f.hessian.fill(0.0);
//    };
//
//    OptimumParams params;
//    params.a = random(m);
//    params.xlower = zeros(n);
//
//    OptimumState state;
//
//    OptimumOptions options;
//    options.mu = 1e-16;
//    options.output.active = true;
////    options.step = StepMode::Conservative;
////    options.kkt.method = SaddlePointMethod::Nullspace;
//    options.kkt.method = SaddlePointMethod::RangespaceDiagonal;
//
//    OptimumSolver solver;
//    solver.setOptions(options);
//    solver.initialize(structure);
//    solver.solve(params, state);
}

