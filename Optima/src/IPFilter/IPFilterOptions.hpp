/*
 * IPFilterOptions.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// Optima includes
#include <Utils/Outputter.hpp>

namespace Optima {

/**
 * The options for the optimisation calculation
 */
struct IPFilterOptions
{
    /**
     * The options for the output of the optimisation calculation
     */
    struct OutputOptions : public Outputter::Options
    {
        /**
         * The boolean flag that indicates if the output of the iterates should be scaled
         */
        bool scaled = false;
    };

    /**
     * The options for the initial guess of the optimisation calculation
     */
    struct InitialGuessOptions
    {
        /**
         * The initial guess for the primal variables @b x in the absence of a proper user-provided one
         */
        double x = 1.0;

        /**
         * The initial guess for the dual variables @b y in the absence of a proper user-provided one
         */
        double y = 0.0;

        /**
         * The initial guess for the dual variables @b z in the absence of a proper user-provided one
         */
        double z = 0.01;

        /**
         * The lower bound on the initial guess of the primal variables @b x
         */
        double xmin = 1.0e-14;

        /**
         * The lower bound on the initial guess of the dual variables @b z
         */
        double zmin = 1.0e-10;
    };

    /**
     * The maximum number of iterations allowed in the algorithm
     */
    unsigned max_iterations = 1000;

    /**
     * The tolerance parameter used for the stopping criteria of the algorithm based on optimality
     */
    double tolerance1 = 1.0e-06;

    /**
     * The tolerance parameter used for the stopping criteria of the algorithm based on step size
     */
    double tolerance2 = 1.0e-9;

    /**
     * The options for the output of the optimisation calculation
     */
    OutputOptions output;

    /**
     * The options for the initial guess of the optimisation calculation
     */
    InitialGuessOptions initialguess;
};

} /* namespace Optima */
