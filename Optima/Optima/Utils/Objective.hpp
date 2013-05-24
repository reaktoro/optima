/*
 * Objective.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <functional>

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

namespace Optima {

/**
 * Defines the result of a objective function evaluation
 *
 * @see ObjectiveFunction
 */
struct ObjectiveResult
{
    /**
     * Constructs a default @ref ObjectiveResult instance
     */
    ObjectiveResult();

    /**
     * Constructs a @ref ObjectiveResult instance
     *
     * This constructor will allocate memory for the
     * data members @ref grad and @ref hessian.
     *
     * @param num_variables The number of variables of the objective function
     */
    ObjectiveResult(unsigned num_variables);

    /**
     * The result of the evaluation of the objective function
     */
    double func;

    /**
     * The result of the evaluation of the gradient of the objective function
     */
    VectorXd grad;

    /**
     * The result of the evaluation of the Hessian of the objective function
     */
    MatrixXd hessian;
};

/**
 * Defines the function signature of a objective function
 *
 * A objective function is here defined as a @c std::function
 * that accepts a vector @a x as argument and return an instance
 * of @ref ObjectiveResult, containing the evaluation of the
 * function as well as its gradient and Hessian.
 *
 * @see ConstraintResult
 */
typedef std::function<
    ObjectiveResult(const VectorXd&)>
        ObjectiveFunction;

} /* namespace Optima */
