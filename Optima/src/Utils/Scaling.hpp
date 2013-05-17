/*
 * Scaling.hpp
 *
 *  Created on: 23 Apr 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// Optima includes
#include <Utils/Constraint.hpp>
#include <Utils/Objective.hpp>

namespace Optima {

class Scaling
{
public:
    /**
     * Constructs a default @ref Scaling instance
     */
    Scaling();

    /**
     * Sets the scaling factors of the variables
     */
    void SetScalingVariables(const VectorXd& Dx);

    /**
     * Sets the scaling factors of the constraint functions
     */
    void SetScalingConstraint(const VectorXd& Dh);

    /**
     * Sets the scaling factor of the objective function
     */
    void SetScalingObjective(double Df);

    /**
     * Gets the scaling factors of the variables
     */
    const VectorXd& GetScalingVariables() const;

    /**
     * Gets the scaling factors of the vector-valued constraint function
     */
    const VectorXd& GetScalingConstraint() const;

    /**
     * Gets the scaling factor of the objective function
     */
    double GetScalingObjective() const;

    /**
     * Checks if the optimisation problem contains scaling factors for the variables
     */
    bool HasScalingVariables() const;

    /**
     * Checks if the optimisation problem contains scaling factors for the constraint functions
     */
    bool HasScalingConstraint() const;

    /**
     *  Checks if the optimisation problem contains a scaling factor for the objective function
     */
    bool HasScalingObjective() const;

    void ScaleX(VectorXd& x) const;

    void ScaleY(VectorXd& y) const;

    void ScaleZ(VectorXd& z) const;

    void ScaleXYZ(VectorXd& x, VectorXd& y, VectorXd& z) const;

    void ScaleConstraint(ConstraintResult& h) const;

    void ScaleObjective(ObjectiveResult& f) const;

    void UnscaleX(VectorXd& x) const;

    void UnscaleY(VectorXd& y) const;

    void UnscaleZ(VectorXd& z) const;

    void UnscaleXYZ(VectorXd& x, VectorXd& y, VectorXd& z) const;

    void UnscaleConstraint(ConstraintResult& h) const;

    void UnscaleObjective(ObjectiveResult& f) const;

private:
        /// The scaling factors of the variables
    VectorXd Dx;

    /// The scaling factors of the contraint functions
    VectorXd Dh;

    /// The scaling factors of the objective function
    double Df;
};

} /* namespace Optima */


