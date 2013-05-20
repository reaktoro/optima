/*
 * Params.hpp
 *
 *  Created on: 5 Apr 2013
 *      Author: allan
 */

#pragma once

namespace Optima {

/**
 * The possible schemes for the calculation of the optimality measure @f$\psi@f$
 */
enum IPFilterPsi
{
    /**
     * Uses the scheme @f$\psi(\mathbf{w})=f(\mathbf{x})+c\mu@f$
     */
    PsiObjective,

    /**
     * Uses the scheme @f$\psi(\mathbf{w})=\mathcal{L}(\mathbf{w})+(c+n)\mu@f$
     */
    PsiLagrange,

    /**
     * Uses the scheme @f$\psi(\mathbf{w})=\lVert\nabla_{x}\mathcal{L}(\mathbf{w})\rVert^{2}+\mathbf{x}^{T}\mathbf{z}/n@f$
     */
    PsiGradLagrange
};

/**
 * The possible schemes for the calculation of the parameter @f$\sigma@f$
 */
enum IPFilterSigma
{
    /**
     * Uses the default IPFilter scheme for the update of the &sigma parameter
     *
     * The update of the parameter &sigma is performed using
     * the following formula:
     *
     * @f[
     *     \sigma=\begin{cases}
     *     \sigma_{\mathrm{fast}} & \mbox{if }\mu<\mu_{\mathrm{threshold}}\\
     *     \sigma_{\mathrm{slow}} & \mbox{otherwise}
     *     \end{cases}
     * @f]
     *
     * where @f$\xi@f$ is given by:
     *
     * @f[
     *     \xi=\frac{\min_{i}\{x_{i}z_{i}\}}{\mathbf{x}^{T}\mathbf{z}/n}.
     * @f]
     */
    SigmaDefault,

    /**
     * Uses the LOQO scheme for the update of the &sigma parameter
     *
     * The update of the parameter &sigma is performed using
     * the following formula:
     *
     * @f[
     *     \sigma=0.1\min\left(0.05\frac{1-\xi}{\xi},2\right)^{3},
     * @f]
     *
     * where the default values of the previous parameter are:
     *
     * @f[
     *     \sigma_{\mathrm{slow}}=10^{-5},\quad\sigma_{\mathrm{fast}}=2.6\cdot10^{-3},\quad\mu_{\mathrm{threshold}}=10^{-6}.
     * @f]
     */
    SigmaLOQO
};

/**
 * The parameters for the optimisation calculation
 */
struct IPFilterParams
{
    /**
     * The parameters for the filter scheme
     */
    struct Filter
    {
        /**
         * The parameter @f$\alpha_{\theta}@f$ used to increase the @f$\theta@f$-borders of the filter region
         *
         * This parameter ensures that a point acceptable by the filter is sufficiently far from
         * its @f$\theta@f$-borders.
         */
        double alpha_theta = 1.0e-03;

        /**
         * The parameter @f$\alpha_{\psi}@f$ used to increase the @f$\psi@f$-borders of the filter region
         *
         * This parameter ensures that a point acceptable by the filter is sufficiently far from
         * its @f$\psi@f$-borders.
         */
        double alpha_psi = 1.0e-03;
    };

    /**
     * The parameters for the inertia correction algorithm
     *
     * These parameters are used in the inertia correction of the KKT matrix whenever its inertia
     * is not <em>(n, m, 0)</em>, where @e n is the number of variables, and @e m is the number of constraints.
     */
    struct InertiaControl
    {
        /**
         * The initial value of the perturbation parameter used in block (1,1) of the KKT matrix
         */
        double epsilon1_initial = 0.5e-06;

        /**
         * The minimum value that the perturbation parameter @c epsilon1 can assume
         */
        double epsilon1_min = 1.0e-20;

        /**
         * The maximum value that the perturbation parameter @c epsilon1 can assume
         */
        double epsilon1_max = 1.0e+06;

        /**
         * The factor used to increase @c epsilon1 on its first increase
         */
        double epsilon1_increase0 = 10.0;

        /**
         * The factor used to increase @c epsilon1
         */
        double epsilon1_increase = 2.0;

        /**
         * The factor used to decrease @c epsilon1
         */
        double epsilon1_decrease = 3.0;

        /**
         * The perturbation used in block (2,2) of the KKT matrix whenever it is singular
         */
        double epsilon2 = 1.0e-20;
    };

    /**
     * The parameters for the central neighbourhood scheme
     */
    struct Neighbourhood
    {
        /**
         * The boolean value that indicates if the neighbourhood search algorithm is activated
         *
         * This boolean value indicates if a seach for a @f$\Delta@f$ that satisfies the neighbourhood
         * conditions should be performed at every iteration.
         *
         * Setting this flag to @c false will cause the calculation to use, at every iteration, the
         * largest possible value of @f$\Delta@f$ that satisfies the positivity conditions of the iterates
         * @b x and @b z.
         */
        bool active = true;

        /**
         * The relaxation parameter used to calculate the initial neighborhood parameter @e M
         *
         * This is the parameter @f$\alpha_{M}^{\circ}@f$ in the algorithm used to calculate the
         * initial neighborhood parameter @f$ M_{0} @f$ as:
         * @f[
         *     M_{0}=\max(M^{\circ},\alpha_{M}^{\circ}\theta(\mathbf{w}_{0})/\mu_{0}).
         * @f]
         */
        double alpha0 = 1.0e+03;

        /**
         * The relaxation parameter used to update the neighborhood parameter @e M
         *
         * This is the parameter @f$\alpha_{M}@f$ in the algorithm used to update the neighborhood
         * parameter @f$ M_{k} @f$ as:
         * @f[
         *     M_{k}=\max(M^{\circ},\alpha_{M}\theta(\mathbf{w}_{k})/\mu_{k}).
         * @f]
         */
        double alpha = 1.0e+01;

        /**
         * The tolerance parameter used to determine the necessity of updating the neighborhood parameter @e M
         *
         * This is the tolerance parameter @f$ \varepsilon_{M} @f$ used in the following condition:
         *
         * @f[
         *     \theta(\mathbf{w}_{k})>\mu_{k}\varepsilon_{M}M_{k},
         * @f]
         *
         * to determine if the neighborhood parameter @f$ M_{k+1} @f$ needs to be updated.
         */
        double epsilon = 1.0e-03;

        /**
         * The maximum allowed value of the neighborhood parameter @e M
         */
        double Mmax = 1.0e+03;

        /**
         * The minimum allowed value of the neighborhood parameter @f$ \gamma @f$
         */
        double gamma_min = 1.0e-03;
    };

    /**
     * The parameters for the calculation of the optimality measure @f$\psi@f$
     */
    struct Psi
    {
        /**
         * The scheme for the calculation of the optimality measure @f$\psi@f$
         */
        IPFilterPsi scheme = PsiObjective;
    };

    /**
     * The parameters for the restart scheme
     */
    struct Restart
    {
        /**
         * The boolean value that indicates if the restart scheme should be used
         */
        bool active = true;

        /**
         * The maximum number of tentatives in the restart scheme
         */
        unsigned tentatives = 4;

        /**
         * The restart factor used to increase @f$ \mu @f$ at every unsuccessful restart tentative
     */
        double factor = 10.0;
    };

    /**
     * The parameters for the restoration algorithm
     */
    struct Restoration
    {
        /**
         * The boolean value that indicates if the restoration algorithm is active
         *
         * If @c true, then whenever the condition:
         *
         * @f[
         *     \theta(\mathbf{w}_{k})>\Delta_{k}\min\{\gamma_{1},\gamma_{2}\Delta_{k}^{\beta}\}
         * @f]
         *
         * is satisfied, the restoration algorithm is started.
         */
        bool active = true;

        /**
         * The parameter @f$\gamma_{1}@f$ in the restoration condition
         */
        double gamma1 = 0.5;

        /**
         * The parameter @f$\gamma_{2}@f$ in the restoration condition
         */
        double gamma2 = 1.0;

        /**
         * The parameter @f$\beta@f$ in the restoration condition
         */
        double beta = 0.75;

        /**
         * The Cauchy parameter @f$\xi_{1}@f$ used in the restoration algorithm to verify convergence progress
         */
        double xi1 = 1.0e-05;

        /**
         * The Cauchy parameter @f$\xi_{2}@f$ used in the restoration algorithm to verify the need of increasing the trust-region radius
         */
        double xi2 = 0.5;
    };

    /**
     * The parameters for the safe tangencial step calculation
     */
    struct SafeTangentialStep
    {
        /**
         * The boolean value that indicates if the safe tangencial step scheme is active
         *
         * If @c true, then whenever @f$ \alpha^{t}(\Delta) < \epsilon_{\alpha_{t}} @f$, where
         * @f$ \epsilon_{\alpha_{t}} @f$ is given by @ref threshold, then a new tangencial step
         * @f$ \mathbf{s}^{t} @f$ is calculated.
         *
         * Note that the KKT coefficient matrix remains unchanged during this process, so its decomposition
         * can be reused.
         */
        bool active = true;

        /**
         * The threshold used to determine if a safe tangencial step calculation is necessary
         *
         * This is the threshold @f$ \epsilon_{\alpha_{t}} @f$ used in @f$ \alpha^{t}(\Delta) < \epsilon_{\alpha_{t}} @f$,
         * to determine if a safe tangencial step is required.
         */
        double threshold = 0.8;
    };

    /**
     * The parameters for the calculation of the centrality parameter @f$\sigma@f$
     */
    struct Sigma
    {
        /**
         * The scheme for the calculation of the centrality parameter @f$\sigma@f$
         */
        IPFilterSigma scheme = SigmaDefault;

        /**
         * The threshold value used to determine the value of @f$\sigma@f$
         *
         * The value of @f$\sigma@f$ is chosen as follows:
         *
         * @f[
         *     \sigma=\begin{cases}
         *         \sigma_{\mathrm{max}} & \mbox{if }\mu<\epsilon_{\mu}\\
         *         \sigma_{\mathrm{min}} & \mbox{otherwise}\end{cases},
         * @f]
         *
         * where @f$\sigma_{\mathrm{min}}@f$ and @f$\sigma_{\mathrm{max}}@f$ are given by @ref main_min and
         * @ref main_max respectively, and @f$\epsilon_{\mu}@f$ is given by @ref threshold_mu.
         */
        double threshold_mu = 1.0e-06;

        /**
         * The threshold used to determine the value of sigma in the safe tangencial step calculation
         *
         * In the safe tangencial step calculation, the value of @f$\sigma@f$ is given by:
         *
         * @f[
         *     \sigma=\begin{cases}
         *         \sigma_{\mathrm{safe,max}} & \mbox{if }\alpha^{t}(\Delta)<\epsilon_{\alpha_{t}}\\
         *         \sigma_{\mathrm{safe,min}} & \mbox{otherwise}\end{cases},
         * @f]
         *
         * where @f$\sigma_{\mathrm{safe,min}}@f$ and @f$\sigma_{\mathrm{safe,max}}@f$ are given by @ref safe_min and
         * @ref safe_max respectively, and @f$\epsilon_{\alpha_{t}}@f$ is given by @ref SafeTangentialStep::threshold.
         */
        double threshold_alphat = 1.0e-3;

        /**
         * The value of @f$\sigma@f$ when @f$\mu@f$ is large
         *
         * See @ref threshold_mu.
         */
        double main_min = 0.1;

        /**
         * The value of @f$\sigma@f$ when @f$\mu@f$ is small
         *
         * See @ref threshold_mu.
         */
        double main_max = 0.5;

        /**
         * The value of @f$\sigma@f$ used in the safe tangencial step calculation when @f$\alpha^{t}(\Delta)@f$ is large
         *
         * See @ref threshold_alphat.
         */
        double safe_min = 0.1;

        /**
         * The value of @f$\sigma@f$ used in the safe tangencial step calculation when @f$\alpha^{t}(\Delta)@f$ is small
         *
         * See @ref threshold_alphat.
         */
        double safe_max = 0.5;

        /**
         * The value of @f$\sigma@f$ used in the restoration algorithm
         */
        double restoration = 1.0;
    };

    /**
     * The parameters for the trust-region algorithm
     */
    struct TrustRegion
    {
        /**
         * The minimum value that the trust-region radius can assume
         */
        double delta_min = 1.0e-12;

        /**
         * The parameter used to increase the trust-region radius
         */
        double delta_increase = 2.0;

        /**
         * The parameter used to decrease the trust-region radius
         */
        double delta_decrease = 0.5;

        /**
         * The initial value of the trust-region radius
         */
        double delta_initial = 1.0e+05;

        /**
         * The parameter used in the verification of the linear model reduction
         *
         * This is the @f$ \kappa @f$ parameter used in the following condition
         * to verify a sufficient decrease in the linear model @f$m_{k}(\mathbf{w}_{k}(\Delta))@f$:
         * @f[
         *     m_{k}(\mathbf{w}_{k})-m_{k}(\mathbf{w}_{k}(\Delta_{k}))<\kappa\theta(\mathbf{w}_{k})^{2}.
         * @f]
         */
        double kappa = 0.1;

        /**
         * The smaller Cauchy condition parameter
         *
         * This is the parameter @f$\eta_{1}@f$ used in the condition:
         *
         * @f[
         *     \rho_{k}\equiv\frac{\psi(\mathbf{w}_{k})-\psi(\mathbf{w}_{k}(\Delta))}{m_{k}(\mathbf{w}_{k})-m_{k}(\mathbf{w}_{k}(\Delta))}>\eta_{1}
         * @f]
         *
         * to determine if @f$\psi@f$ has achieved a sufficient decrease with respect to the linear model
         * @f$ m_{k}(\mathbf{w}_{k}(\Delta)) @f$.
         */
        double eta1 = 1.0e-04;

        /**
         * The larger Cauchy condition parameter
         *
         * This is the parameter @f$\eta_{2}@f$ used in the condition:
         *
         * @f[
         *     \rho_{k}\equiv\frac{\psi(\mathbf{w}_{k})-\psi(\mathbf{w}_{k}(\Delta))}{m_{k}(\mathbf{w}_{k})-m_{k}(\mathbf{w}_{k}(\Delta))}>\eta_{2}
         * @f]
         *
         * to determine if the trust-region radius @f$ \Delta @f$ needs to be increased.
         */
        double eta2 = 0.8;
    };

    /// The parameters for the filter scheme
    Filter filter;

    /// The parameters for the inertia correction algorithm
    InertiaControl inertia;

    /// The parameters for the central neighbourhood scheme
    Neighbourhood neighbourhood;

    /// The parameters for the calculation of the optimality measure @f$\psi@f$
    Psi psi;

    /// The parameters for the restart scheme
    Restart restart;

    /// The parameters for the restoration algorithm
    Restoration restoration;

    /// The parameters for the safe tangencial step calculation
    SafeTangentialStep safestep;

    /// The parameters for the calculation of the centrality parameter @f$\sigma@f$
    Sigma sigma;

    /// The parameters for the trust-region algorithm
    TrustRegion main;
};

} /* namespace Optima */
