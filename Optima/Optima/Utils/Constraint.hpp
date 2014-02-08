/*
 * Constraint.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <functional>
#include <vector>

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

namespace Optima {

/**
 * Defines the result of a constraint function evaluation
 *
 * @see ConstraintFunction
 */
struct ConstraintResult
{
    /**
     * Constructs a default @ref ConstraintResult instance
     */
    ConstraintResult();

    /**
     * Constructs a @ref ConstraintResult instance
     *
     * This constructor will allocate memory for the data
     * members @ref func and @ref grad.
     *
     * Note that this constructor does not allocate memory
     * for the data member @c hessian, because not rarely the
     * Hessian of the constraint function is zero or neglected.
     *
     * @param num_variables The number of variables of the constraint function
     * @param num_constraints The number of constraints of the constraint function
     */
    ConstraintResult(unsigned num_variables, unsigned num_constraints);

    /**
     * Constructs a @ref ConstraintResult instance
     *
     * This constructor will allocate memory for the data
     * members @ref func and @ref grad. Moreover, if the boolean value
     * @c allocate_hessian is @c true, than the data member @ref hessian
     * will also have memory allocated.
     *
     * @param num_variables The number of variables of the constraint function
     * @param num_constraints The number of constraints of the constraint function
     * @param allocate_hessian The boolean value that indicates if memory will be allocated for @c hessian
     */
    ConstraintResult(unsigned num_variables, unsigned num_constraints, bool allocate_hessian);

    /**
     * The result of the evaluation of the constraint function
     */
    VectorXd func;

    /**
     * The result of the evaluation of the gradient of the constraint function
     */
    MatrixXd grad;

    /**
     * The result of the evaluation of the Hessian of the constraint functions
     *
     * The Hessian of the vector function <i>h(x)</i> defining the constraint function
     * is here represented by a @c std::vector of @a n by @a n matrices, where @a n
     * is the dimension of @a x. Each matrix is the Hessian of a component of the
     * constraint function <i>h(x)</i>. For instance, the @a i-th matrix is the Hessian
     * of the scalar function <i>h<sub>i</sub>(x)</i>.
     */
    std::vector<MatrixXd> hessian;
};

/**
 * Defines the function signature of a constraint function
 *
 * A constraint function is here defined as a @c std::function
 * that accepts a vector as argument and return an instance
 * of @ref ConstraintResult, containing the evaluation of the
 * function as well as its gradient and Hessian.
 *
 * @see ConstraintResult
 */
typedef std::function<
    ConstraintResult(const VectorXd&)>
        ConstraintFunction;

} /* namespace Optima */
