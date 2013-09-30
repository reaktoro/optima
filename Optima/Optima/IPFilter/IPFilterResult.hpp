/*
 * IPFilterResult.hpp
 *
 *  Created on: 16 Apr 2013
 *      Author: allan
 */

#pragma once

namespace Optima {

/**
 * The result information of the calculation performed by the IPFilter algorithm
 *
 * @see IPFilterSolver
 */
struct IPFilterResult
{
    /**
     * The boolean flag that indicates if the calculation converged
     */
    bool converged = false;

    /**
     * The number of iterations performed in the calculation
     */
    unsigned num_iterations = 0;

    /**
     * The number of evaluations of the objective function performed in the calculation
     */
    unsigned num_objective_evals = 0;

    /**
     * The number of evaluations of the constraint function performed in the calculation
     */
    unsigned num_constraint_evals = 0;

    /**
     * The number of times the calculation entered the restoration phase algorithm
     */
    unsigned num_restorations = 0;

    /**
     * The conversion operator that returns true if the calculation converged to a local solution
     */
    operator bool() { return converged; }
};

} /* namespace Optima */
