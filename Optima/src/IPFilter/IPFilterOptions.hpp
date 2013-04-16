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
 * Defines the types of schemes used for the calculation of the optimality measure \f$\psi\f$
 */
enum IPFilterPsiScheme
{
    /// Uses the scheme \f$\psi(\mathbf{w})=f(\mathbf{x})+c\mu\f$
    Objective,

    /// Uses the scheme \f$\psi(\mathbf{w})=\mathcal{L}(\mathbf{w})+(c+n)\mu\f$
    Lagrange,

    /// Uses the scheme \f$\psi(\mathbf{w})=\lVert\nabla_{x}\mathcal{L}(\mathbf{w})\rVert^{2}+\mathbf{x}^{T}\mathbf{z}/n\f$
    GradLagrange
};

/**
 * The options used for the algorithm
 */
struct IPFilterOptions
{
    /**
     * The scheme used for the calculation of the optimality measure \f$\psi\f$
     *
     * @see IPFilterPsiScheme
     */
    IPFilterPsiScheme psi = Objective;

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
    Outputter::Options output;
};

} /* namespace Optima */
