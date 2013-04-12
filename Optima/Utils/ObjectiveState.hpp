/*
 * ObjectiveState.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

namespace Optima {

struct ObjectiveState
{
    ObjectiveState()
    {}

    ObjectiveState(unsigned dimx)
    : grad(dimx), hessian(dimx, dimx) {}

    double func;

    VectorXd grad;

    MatrixXd hessian;
};

} /* namespace Optima */
