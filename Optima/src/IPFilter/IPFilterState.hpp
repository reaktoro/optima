/*
 * IPFilterState.hpp
 *
 *  Created on: 11 Apr 2013
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

/**
 * The algorithmic state at the point (x,y,z)
 */
struct IPFilterState
{
    /// The iterates x, y, z of the algorithm
    VectorXd x, y, z;

    /// The state of the objective function at x
    ObjectiveResult f;

    /// The state of the constraint function at x
    ConstraintResult h;

    /// The barrier parameter at (x,z)
    double mu;

    /// The optimality measures theta_h, theta_c, and theta_l
    double thh, thc, thl;

    /// The optimality measure theta
    double theta;

    /// The optimality measure psi
    double psi;

    /// The feasibility, centrality, and optimality errors respectively
    double errorh, errorc, errorl;

    /// The maximum among the feasibility, centrality, and optimality errors
    double error;
};

} /* namespace Optima */


