/*
 * Functions.hpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// Optima includes
#include <Utils/Math.hpp>

namespace Optima {

inline double CalculateLargestBoundaryStep(const VectorXd& p, const VectorXd& dp)
{
    double step = INF;
    for(unsigned i = 0; i < p.rows(); ++i)
    {
        const double aux = -p[i]/dp[i];
        if(aux > 0.0) step = std::min(step, aux);
    }
    return step;
}

} /* namespace Optima */


