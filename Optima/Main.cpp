/*
 * Main.cpp
 *
 *  Created on: 22 Mar 2013
 *      Author: allan
 */

// C++ includes
#include <functional>
#include <iostream>
#include <tuple>

// Optima includes
#include <IPFilter/AlgorithmIPFilter.hpp>
using namespace Optima;

ObjectiveState Obj1(const VectorXd& x)
{
    ObjectiveState f;

    f.func    = x[0]*x[0] + x[1]*x[1];
    f.grad    = 2.0*x;
    f.hessian = 2.0*MatrixXd::Identity(2, 2);

    return f;
}

ConstraintState Cons1(const VectorXd& x)
{
    ConstraintState h(2, 1);

    h.func << x[0] - x[1];
    h.grad << 1.0, -1.0;

    return h;
}

int main()
{
    OptimumProblem problem;
    problem.dim_objective  = 2;
    problem.dim_constraint = 1;
    problem.objective      = Obj1;
    problem.constraint     = Cons1;

    IPFilter::AlgorithmIPFilter solver;

    solver.SetProblem(problem);

    VectorXd x(2);
    x << 2, 3;

    solver.Solve(x);

    std::cout << x << std::endl;

}
