/*
 * Constraint.cpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#include "Constraint.hpp"

namespace Optima {

ConstraintResult::ConstraintResult()
{}

ConstraintResult::ConstraintResult(unsigned num_variables, unsigned num_constraints)
: func(num_constraints), grad(num_constraints, num_variables)
{}

ConstraintResult::ConstraintResult(unsigned num_variables, unsigned num_constraints, bool allocate_hessian)
: func(num_constraints), grad(num_constraints, num_variables)
{
    if(allocate_hessian)
        hessian.resize(num_constraints,
            MatrixXd(num_variables, num_variables));
}

} /* namespace Optima */
