/*
 * ConstraintState.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

namespace Optima {

struct ConstraintState
{
    ConstraintState()
    {}

    ConstraintState(unsigned dimx, unsigned dimh)
    : func(dimh), grad(dimh, dimx) {}

    VectorXd func;

    MatrixXd grad;

    std::vector<MatrixXd> hessian;
};

} /* namespace Optima */
