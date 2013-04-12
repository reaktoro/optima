/*
 * Functions.hpp
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

// Optima includes
#include <Utils/ObjectiveState.hpp>
#include <Utils/ConstraintState.hpp>

namespace Optima {

typedef std::function<ObjectiveState(const VectorXd&)>
    ObjectiveFunction;

typedef std::function<ConstraintState(const VectorXd&)>
    ConstraintFunction;

} /* namespace Optima */


