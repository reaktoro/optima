/*
 * OptimumProblem.cpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#include "OptimumProblem.hpp"

namespace Optima {

OptimumProblem::OptimumProblem()
: num_variables(0), num_constraints(0), has_constraints(false)
{}

void OptimumProblem::SetNumConstraints(unsigned num_constraints)
{
    this->num_constraints = num_constraints;
}

void OptimumProblem::SetNumVariables(unsigned num_variables)
{
    this->num_variables = num_variables;
}

bool OptimumProblem::HasConstraints() const
{
    return has_constraints;
}

unsigned OptimumProblem::GetNumConstraints() const
{
    return num_constraints;
}

unsigned OptimumProblem::GetNumVariables() const
{
    return num_variables;
}

void OptimumProblem::SetConstraintFunction(const ConstraintFunction& constraint)
{
    this->constraint = constraint;

    has_constraints = true;
}

void OptimumProblem::SetObjectiveFunction(const ObjectiveFunction& objective)
{
    this->objective = objective;
}

ConstraintResult OptimumProblem::Constraint(const VectorXd& x) const
{
    return constraint(x);
}

ObjectiveResult OptimumProblem::Objective(const VectorXd& x) const
{
    return objective(x);
}

} /* namespace Optima */
