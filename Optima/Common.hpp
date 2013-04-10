/*
 * Common.hpp
 *
 *  Created on: 30 Jan 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <functional>
#include <tuple>
#include <vector>

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

namespace Optima {

/**
 * Defines a function type for objective functions of optimisation problem
 *
 * @param x The vector where the function is evaluated
 *
 * @return A tuple (f, grad, hessian) containing the function evaluation @a f,
 * its gradient vector @a grad, and its hessian matrix @a hessian
 */
typedef std::function<
	std::tuple<double, VectorXd, MatrixXd>
		(const VectorXd&)>
			Objective;

/**
 * Defines the constraint vector function of the minimisation problem
 *
 * @param x The vector where the function is evaluated
 *
 * @return A tuple (h, jac, hessians) containing the function evaluation @a h,
 * its jacobian matrix @a jac, and the hessian matrices @a hessians of each constraint component
 */
typedef std::function<
	std::tuple<VectorXd, MatrixXd, std::vector<MatrixXd>>
		(const VectorXd&)>
			Constraint;

struct ObjectiveState
{
    ObjectiveState()
    {}

    ObjectiveState(unsigned dimx)
    : grad(dimx), hessian(dimx, dimx) {}

    double func;

    VectorXd grad;

    MatrixXd hessian;
};

struct ConstraintState
{
    ConstraintState()
    {}

    ConstraintState(unsigned dimx, unsigned dimh)
    : func(dimh), grad(dimh, dimx) {}

    VectorXd func;

    MatrixXd grad;

    std::vector<MatrixXd> hessian;
};

typedef std::function<ObjectiveState(const VectorXd&)>
    ObjectiveFunction;

typedef std::function<ConstraintState(const VectorXd&)>
    ConstraintFunction;

struct OptimumProblem
{
    unsigned dim_objective;

    unsigned dim_constraint;

    ObjectiveFunction objective;

    ConstraintFunction constraint;
};

} /* namespace Optima */

