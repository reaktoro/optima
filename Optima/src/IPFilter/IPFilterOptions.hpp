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
 * Defines the possible schemes for the calculation of the optimality measure &psi
 */
enum IPFilterPsi
{
    /**
     * Uses the scheme \f$\psi(\mathbf{w})=f(\mathbf{x})+c\mu\f$
     */
    PsiObjective,

    /**
     * Uses the scheme \f$\psi(\mathbf{w})=\mathcal{L}(\mathbf{w})+(c+n)\mu\f$
     */
    PsiLagrange,

    /**
     * Uses the scheme \f$\psi(\mathbf{w})=\lVert\nabla_{x}\mathcal{L}(\mathbf{w})\rVert^{2}+\mathbf{x}^{T}\mathbf{z}/n\f$
     */
    PsiGradLagrange
};

/**
 * Defines the possible schemes for the calculation of the parameter &sigma
 */
enum IPFilterSigma
{
    /**
     * Uses the default IPFilter scheme for the update of the &sigma parameter
     *
     * The update of the parameter &sigma is performed using
     * the following formula:
     *
     * \f[
     *     \sigma=\begin{cases}
     *     \sigma_{\mathrm{fast}} & \mbox{if }\mu<\mu_{\mathrm{threshold}}\\
     *     \sigma_{\mathrm{slow}} & \mbox{otherwise}
     *     \end{cases}
     * \f]
     *
     * where \f$\xi\f$ is given by:
     *
     * \f[
     *     \xi=\frac{\min_{i}\{x_{i}z_{i}\}}{\mathbf{x}^{T}\mathbf{z}/n}.
     * \f]
     */
    SigmaDefault,

    /**
     * Uses the LOQO scheme for the update of the &sigma parameter
     *
     * The update of the parameter &sigma is performed using
     * the following formula:
     *
     * \f[
     *     \sigma=0.1\min\left(0.05\frac{1-\xi}{\xi},2\right)^{3},
     * \f]
     *
     * where the default values of the previous parameter are:
     *
     * \f[
     *     \sigma_{\mathrm{slow}}=10^{-5},\quad\sigma_{\mathrm{fast}}=2.6\cdot10^{-3},\quad\mu_{\mathrm{threshold}}=10^{-6}.
     * \f]
     */
    SigmaLOQO
};

/**
 * The options used for the algorithm
 */
struct IPFilterOptions
{
    /**
     * The scheme used for the calculation of the optimality measure &psi
     *
     * @see IPFilterPsi
     */
    IPFilterPsi psi = PsiObjective;

    /**
     * The scheme used for the calculation of the parameter &sigma
     *
     * @see IPFilterSigma
     */
    IPFilterSigma sigma = SigmaDefault;

    /**
     * The maximum number of iterations allowed in the algorithm
     */
    unsigned max_iterations = 1000;

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
     * The boolean value that activates output during the calculation
     */
    Outputter::Options output;

    bool output_scaled = false;
};

} /* namespace Optima */
