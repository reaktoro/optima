/*
 * Scaling.hpp
 *
 *  Created on: 30 Jan 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

namespace Optima {

class Scaling
{
public:
	/**
	 * Constructs a @ref Scaling instance with given number of variables and constraints
	 *
	 * @param num_variables The number of variables in the optimisation problem
	 * @param num_constraints The number of constraints in the optimisation problem
	 */
	Scaling(unsigned num_variables, unsigned num_constraints);

	/**
	 * Sets the scaling value used to scale the objective function
	 *
	 * @param cf The scaling value for the objective function
	 */
	void SetScaleObjective(double cf);

	/**
	 * Sets the scaling diagonal vector used to scale the constraint function
	 *
	 * @param ch The scaling diagonal vector used to scale the constraint function
	 */
	void SetScaleConstraint(const VectorXd& ch);

	/**
	 * Sets the diagonal scaling matrix used to scale the variables
	 *
	 * @param cx The diagonal scaling matrix for the variables
	 */
	void SetScaleVariables(const VectorXd& cx);

	/**
	 * Gets the scaling value used to scale the objective function
	 */
	double GetScaleObjective() const;

	/**
	 * Gets the scaling diagonal vector used to scale the constraint function
	 */
	const VectorXd& GetScaleConstraint() const;

	/**
	 * Gets the diagonal scaling matrix used to scale the variables
	 */
	const VectorXd& GetScaleVariables() const;

	/**
	 * Scales the vector of variables @c x
	 *
	 * @param x The vector of unscaled variables to be scaled
	 * @return A vector containing the scaled variables
	 */
	VectorXd Scale(const VectorXd& x) const;

	/**
	 * Unscales the vector of variables @c x
	 *
	 * @param x The vector of scaled variables to be unscaled
	 * @return A vector containing the unscaled variables
	 */
	VectorXd Unscale(const VectorXd& x) const;

private:
	/// The number of variables in the optimisation problem
	unsigned num_variables;

	/// The number of constraints in the optimisation problem
	unsigned num_constraints;

	/// The scaling value used to scale the objective function (default: 1.0)
	double cf;

	/// The scaling diagonal vector used to scale the constraint function (default: ones)
	VectorXd ch;

	/// The scaling diagonal vector used to scale the variables (default: ones)
	VectorXd cx;
};

/**
 * Generates a scaled version of the objective function
 *
 * Note that this function creates a copy of the objective function. If
 * this instance is large, it might be more adequate and efficient to use
 * the @ref ScaledObjectiveRef function. This will use a reference of the
 * objective function @c objective instead.
 *
 * @param objective The objective function to be scaled
 * @param scaling The scaling parameters
 *
 * @return The scaled objective function
 */
Objective ScaledObjective(const Objective& objective, const Scaling& scaling);

/**
 * Generates a scaled version of the objective function
 *
 * This function uses a reference of @c objective instead of creating a copy as
 * @ref ScaledObjective does. It should be used when that instance is large,
 * therefore, improving efficiency. Note that @c objective is expected to not go
 * out of scope, otherwise an undefined behaviour will happen.
 *
 * @param objective The objective function to be scaled
 * @param scaling The scaling parameters
 *
 * @return The scaled objective function
 */
Objective ScaledObjectiveRef(const Objective& objective, const Scaling& scaling);

/**
 * Generates a scaled version of the constraint function
 *
 * Note that this function creates a copy of the constraint function. If
 * this instance is large, it might be more adequate and efficient to use
 * the @ref ScaledConstraintRef function. This will use a reference of the
 * constraint function @c constraint instead.
 *
 * @param constraint The constraint function to be scaled
 * @param scaling The scaling parameters
 *
 * @return The scaled constraint function
 */
Constraint ScaledConstraint(const Constraint& constraint, const Scaling& scaling);

/**
 * Generates a scaled version of the constraint function
 *
 * This function uses a reference of @c constraint instead of creating a copy as
 * @ref ScaledConstraint does. It should be used when that instance is large,
 * therefore, improving efficiency. Note that @c constraint is expected to not go
 * out of scope, otherwise an undefined behaviour will happen.
 *
 * @param constraint The constraint function to be scaled
 * @param scaling The scaling parameters
 *
 * @return The scaled constraint function
 */
Constraint ScaledConstraintRef(const Constraint& constraint, const Scaling& scaling);

} /* namespace Optima */
