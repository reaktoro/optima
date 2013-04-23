/*
 * IPFilterResult.hpp
 *
 *  Created on: 16 Apr 2013
 *      Author: allan
 */

#pragma once

namespace Optima {

/**
 * Defines the result of the calculation performed by the IPFilter algorithm
 */
struct IPFilterResult
{
    /**
     * The number of iterations executed in the IPFilter algorithm
     */
    unsigned iterations = 0;

    /**
     * The number of evaluations of the objective function
     */
    unsigned objective_evals = 0;

    /**
     * The number of evaluations of the constraint function
     */
    unsigned constraint_evals = 0;

    /**
     * The number of times the calculation entered the restoration phase algorithm
     */
    unsigned restorations = 0;

    /**
     * The boolean value that indicates if the calculation converged to a local solution
     */
    bool converged = false;

    /**
     * The error message if the last calculation failed.
     */
    std::string error;

    /**
     * The conversion operator that returns true if the calculation converged to a local solution
     */
    operator bool() { return converged; }
};

} /* namespace Optima */


