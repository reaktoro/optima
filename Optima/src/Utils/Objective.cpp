/*
 * ObjectiveResult.cpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#include "Objective.hpp"

namespace Optima {

ObjectiveResult::ObjectiveResult()
: func(0)
{}

ObjectiveResult::ObjectiveResult(unsigned num_variables)
: grad(num_variables), hessian(num_variables, num_variables)
{}

} /* namespace Optima */
