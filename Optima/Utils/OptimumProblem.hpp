#include <Utils/Functions.hpp>
/*
 * OptimumProblem.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// Optima includes
#include <Utils/Functions.hpp>

namespace Optima {

/**
 * The definition of the optimisation problem
 */
struct OptimumProblem
{
    /// The number of variables in the optimisation problem
    unsigned num_variables;

    /// The number of equality constraints in the optimisation problem
    unsigned num_constraints;

    /// The objective function in the optimisation problem
    ObjectiveFunction objective;

    /// The equality constraint function in the optimisation problem
    ConstraintFunction constraint;
};

} /* namespace Optima */
