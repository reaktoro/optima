///*
// * Main.cpp
// *
// *  Created on: 22 Mar 2013
// *      Author: allan
// */
//
//// C++ includes
//#include <functional>
//#include <iostream>
//#include <tuple>
//
//// Optima includes
//#include <Optima/IPFilter/IPFilterSolver.hpp>
//using namespace Optima;
//
//ObjectiveResult Obj1(const VectorXd& x)
//{
//    ObjectiveResult f;
//
//    f.func    = x[0]*x[0] + x[1]*x[1];
//    f.grad    = 2.0*x;
//    f.hessian = 2.0*MatrixXd::Identity(2, 2);
//
//    return f;
//}
//
//ConstraintResult Cons1(const VectorXd& x)
//{
//    ConstraintResult h(2, 1);
//
//    h.func << x[0] - x[1];
//    h.grad << 1.0, -1.0;
//
//    return h;
//}
//
//int main()
//{
//    OptimumProblem problem;
//    problem.SetNumVariables(2);
//    problem.SetNumConstraints(1);
//    problem.SetObjectiveFunction(Obj1);
//    problem.SetConstraintFunction(Cons1);
//
//    IPFilterSolver::Options options;
//    options.output = true;
//
//    IPFilterSolver solver;
//
//    solver.SetOptions(options);
//    solver.SetProblem(problem);
//
//    VectorXd x(2);
//    x << 200, 30000;
//
//    solver.Solve(x);
//}
