/*
 * IPFilterOptions.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

namespace Optima {

/**
 * The options used for the algorithm
 */
struct IPFilterOptions
{
    /**
     * The scheme used for the calculation of the psi optimality measure
     */
    unsigned psi_scheme = 0;

    /**
     * The maximum number of iterations allowed in the algorithm
     */
    unsigned max_iter = 100;

    /**
     * The start value used for the barrier parameter
     *
     * This parameter is used when only the iterate x is
     * provided as initial guess. In that case, it is
     * necessary an initial value of \f$ \mu \f$ in order
     * to estimate an initial guess for the Lagrange
     * multipliers y and z.
     */
    double mu = 0.1;

    /**
     * The tolerance parameter used for the stopping criteria of the algorithm
     */
    double tolerance = 1.0e-06;

    /**
     * The logical flag that activates output during the calculation
     */
    bool output = false;
};

} /* namespace Optima */
