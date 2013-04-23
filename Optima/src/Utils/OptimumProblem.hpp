/*
 * OptimumProblem.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// Optima includes
#include <Utils/Constraint.hpp>
#include <Utils/Objective.hpp>

namespace Optima {

/**
 * Defines the optimisation problem
 *
 * This class is used to define an optimisation problem.
 * Both number of variables and constraints are provided here,
 * as well as the objective and contraint functions.
 */
class OptimumProblem
{
public:
    /**
     * Constructs a default @ref OptimumProblem instance
     */
    OptimumProblem();

    /**
     * Sets the number of constraints in the optimisation problem
     */
    void SetNumConstraints(unsigned num_constraints);

    /**
     * Sets the number of variables in the optimisation problem
     */
    void SetNumVariables(unsigned num_variables);

    /**
     * Sets the constraint function of the optimisation problem
     */
    void SetConstraintFunction(const ConstraintFunction& constraint);

    /**
     * Sets the objective function of the optimisation problem
     */
    void SetObjectiveFunction(const ObjectiveFunction& objective);

    /**
     * Checks if the optimisation problem contains constraints
     */
    bool HasConstraints() const;

    /**
     * Gets the number of constraints of the optimisation problem
     */
    unsigned GetNumConstraints() const;

    /**
     * Gets the number of variables of the optimisation problem
     */
    unsigned GetNumVariables() const;

    /**
     * Evaluates the constraint function
     *
     * @param x The vector of variables
     * @return An instance of @ref ConstraintResult
     */
    ConstraintResult Constraint(const VectorXd& x) const;

    /**
     * Evaluates the objective function
     *
     * @param x The vector of variables
     * @return An instance of @ref ObjectiveResult
     */
    ObjectiveResult Objective(const VectorXd& x) const;

private:
    /// The number of variables in the optimisation problem
    unsigned num_variables;

    /// The number of equality constraints in the optimisation problem
    unsigned num_constraints;

    /// The boolean value that indicates that a contraint function has been provided
    bool has_constraints;

    /// The objective function in the optimisation problem
    ObjectiveFunction objective;

    /// The equality constraint function in the optimisation problem
    ConstraintFunction constraint;
};

} /* namespace Optima */
