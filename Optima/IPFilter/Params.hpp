/*
 * Params.hpp
 *
 *  Created on: 5 Apr 2013
 *      Author: allan
 */

#pragma once

namespace Optima {
namespace IPFilter {

/**
 * The list of algorithm parameters and their default values
 */
struct Params
{
    //===========================
    // MAIN ALGORITHM PARAMETERS
    //===========================
    /**
     * The minimum value that the trust-region radius can assume
     */
    double delta_min = 1.0e-12;

    /**
     * The parameter used to increase the trust-region radius when it becomes allowed
     */
    double delta_increase_factor = 2.0;

    /**
     * The parameter used to decrease the size of the trust-region
     */
    double delta_decrease_factor = 0.5;

    /**
     * The parameter used in the verification of the linear model reduction
     *
     * This is the \f$ \kappa \f$ parameter used in the following condition
     * to verify a sufficient decrease in the linear model \f$ m(\mathbf{w}(\Delta)) \f$:
     * \f[
     *     m_{k}(\mathbf{w}_{k})-m_{k}(\mathbf{w}_{k}(\Delta_{k}))<\kappa\theta(\mathbf{w}_{k})^{2}.
     * \f]
     */
    double kappa = 0.1;

    /**
     * The smaller Cauchy condition parameter
     *
     * This is the parameter \f$ eta_{1} \f$ used in the following condition
     * to determine if \f$ \theta_{g} \f$ has achieved a sufficient decrease
     * with respect to the linear model \f$ m(\mathbf{w}(\Delta)) \f$:
     * \f[
     *     \rho_{k}\equiv\frac{\theta_{g}(\mathbf{w}_{k})-\theta_{g}(\mathbf{w}_{k}(\Delta_{k}))}{m(\mathbf{w}_{k})-m(\mathbf{w}_{k}(\Delta_{k}))}\geq\eta_{1}
     * \f]
     */
    double eta_small = 1.0e-04;

    /**
     * The larger Cauchy condition parameter
     *
     * This is the parameter \f$ eta_{2} \f$ used in the following condition
     * to determine if the trust-region radius \f$ \Delta \f$ needs to be increased:
     * \f[
     *     \rho_{k}\equiv\frac{\theta_{g}(\mathbf{w}_{k})-\theta_{g}(\mathbf{w}_{k}(\Delta_{k}))}{m(\mathbf{w}_{k})-m(\mathbf{w}_{k}(\Delta_{k}))}\geq\eta_{2}
     * \f]
     */
    double eta_large = 0.8;

    /**
     * The parameter used in the resetting of the Lagrange multipliers z
     *
     */
    double kappa_zreset = 1.0e+08;

    //==================================
    // RESTORATION ALGORITHM PARAMETERS
    //==================================
    /**
     * The parameter used to verify the necessity of the restoration algorithm
     *
     * This is the parameter \f$ \gamma_{1} \f$ in the following condition
     * that indicates the necessity of the restoration algorithm:
     * \f[
     *     \theta(\mathbf{w})>\Delta_{k}\min\{\gamma_{1},\gamma_{2}\Delta_{k}^{\beta}\}.
     * \f]
     */
    double gamma1 = 0.5;

    /**
     * The parameter used to verify the necessity of the restoration algorithm
     *
     * This is the parameter \f$ \gamma_{2} \f$ in the following
     * condition that indicates the necessity of the restoration algorithm:
     * \f[
     *     \theta(\mathbf{w})>\Delta_{k}\min\{\gamma_{1},\gamma_{2}\Delta_{k}^{\beta}\}.
     * \f]
     */
    double gamma2 = 1.0;

    /**
     * The parameter used to verify the necessity of the restoration algorithm
     *
     * This is the parameter \f$ \beta \f$ in the following
     * condition that indicates the necessity of the restoration algorithm:
     * \f[
     *     \theta(\mathbf{w})>\Delta_{k}\min\{\gamma_{1},\gamma_{2}\Delta_{k}^{\beta}\}.
     * \f]
     */
    double beta = 0.75;

    double xi1 = 1.0e-05;

    double xi2 = 0.5;

    //============================================================
    // INERTIAL CORRECTION AND DIAGONAL REGULARIZATION PARAMETERS
    //============================================================
    /**
     * The initial value of the perturbation parameter used in block (1,1) of the KKT matrix
     *
     * This perturbation value is used whenever the
     * inertia of the KKT matrix is not (n,m,0).
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
    double epsilon1_increase_factor_first = 10.0;

    /**
     * The factor used to increase @c epsilon1
     */
    double epsilon1_increase_factor = 2.0;

    /**
     * The factor used to decrease @c epsilon1
     */
    double epsilon1_decrease_factor = 3.0;

    /**
     * The perturbation used in block (2,2) of the KKT matrix whenever it is singular
     */
    double epsilon2 = 1.0e-20;

    //============================
    // TANGENTIAL STEP PARAMETERS
    //============================
    /**
     * The logical flag that indicates if safe tangencial step calculations are performed when necessary
     *
     * If @c safe_step_active is true, then whenever the
     * tangencial step length @c alphat < @c safe_step_threshold,
     * a new tangencial step @c st is calculated.
     *
     * Note that the KKT coefficient matrix remains
     * unchanged during this process, so its decomposition
     * can be reused.
     */
      bool safe_step_active = true;

    /**
     * The threshold used to determine if a safe tangencial step calculatio is necessary
     *
     * If @c alphat < @c safe_step_threshold, then
     * recompute the tangencial step by using a different
     * value for @c sigma.
     */
    double safe_step_threshold = 0.8;

    /**
     * The threshold used to determine which value of sigma is used in the safe tangencial step calculation
     *
     * In the safe tangencial step calculation, if
     * @c alphat < @c safe_step_threshold_alphat, then
     * @c sigma = @c sigma_safe_max, else @c sigma = @c sigma_safe_min.
     */
    double safe_step_threshold_alphat = 1.0e-3;

    /**
     * The threshold value of @c mu that indicates which value of @c sigma to be used
     *
     * The value of @c sigma is chosen as follows: if @c mu < @c mu_threshold, then
     * @c sigma = @c sigma_fast, else
     * @c sigma = @c sigma_slow.
     */
    double mu_threshold = 1.0e-06;

    /**
     * The value of @c sigma used when @c mu is large
     *
     * This value of sigma is used whenever @c mu > @c mu_threshold.
     */
    double sigma_slow = 1.0e-05;

    /**
     * The value of @c sigma used when @c mu is small
     *
     * This value of sigma is used whenever @c mu < @c mu_threshold.
     */
    double sigma_fast = 2.6e-03;

    /**
     * The minimum value of @c sigma used in the safe tangencial step calculation
     *
     * The safe tangencial calculation uses this minimum
     * value of @c sigma when @c alphat > @c safe_step_threshold_alphat.
     */
    double sigma_safe_min = 1.0e-03;

    /**
     * The maximum value of @c sigma used in the safe tangencial step calculation
     *
     * The safe tangencial calculation uses this maximum
     * value of @c sigma when @c alphat < @c safe_step_threshold_alphat.
     */
    double sigma_safe_max = 0.1;

    /**
     * The value of @c sigma used during the restoration algorithm
     */
    double sigma_restoration = 1.0;

    //=========================
    // NEIGHBORHOOD PARAMETERS
    //=========================
    /**
     * The relaxation parameter used to update the neighborhood parameter M
     *
     * This is the parameter \f$ \alpha_{M} \f$ in the algorithm used to
     * update the neighborhood parameter M as:
     * \f[
     *     M=\max(M_{0},\alpha_{M}(\theta_{h}(\mathbf{w}_{k})+\theta_{\mathcal{L}}(\mathbf{w}_{k}))/\mu_{k}).
     * \f]
     */
    double alphaM = 1.0e+01;

    /**
     * The relaxation parameter used to calculate the initial neighborhood parameter M
     *
     * This is the parameter \f$ \alpha_{M}^{\circ} \f$ in the algorithm used to
     * calculate the initial neighborhood parameter M as:
     * \f[
     *     M_{0}=\max(M^{\circ},\alpha_{M}^{\circ}(\theta_{h}(\mathbf{w}_{0})+\theta_{\mathcal{L}}(\mathbf{w}_{0}))/\mu_{k}).
     * \f]
     */
    double alphaM_initial = 1.0e+03;

    /**
     * The tolerance parameter used to determine the necessity of updating the neighborhood parameter M
     *
     * This is the tolerance parameter \f$ \varepsilon_{M} \f$ used in the algorithm
     * to determine if the neighborhood parameter M needs to be updated. It appears in
     * the following condition:
     * \f[
     *     \theta_{h}(\mathbf{w}_{k})+\theta_{\mathcal{L}}(\mathbf{w}_{k})>\mu_{k}\varepsilon_{M}M.
     * \f]
     */
    double epsilonM = 1.0e-03;

    /**
     * The maximum allowed value of the neighborhood parameter M
     */
    double neighM_max = 1.0e+03;

    /**
     * The minimum allowed value of the neighborhood parameter \f$ \gamma \f$
     */
    double gamma_min = 1.0e-03;

    //===================
    // FILTER PARAMETERS
    //===================
    /**
     * The parameter used to increase the \f$\theta\f$-borders of the filter region
     *
     * This parameter ensures that a point acceptable by the
     * filter is sufficiently far from its \f$\theta\f$-borders.
     */
    double alpha_theta = 1.0e-03;

    /**
     * The parameter used to increase the \f$\psi\f$-borders of the filter region
     *
     * This parameter ensures that a point acceptable by the
     * filter is sufficiently far from its \f$\psi\f$-borders.
     */
    double alpha_psi = 1.0e-03;
};

} /* namespace IPFilter */
} /* namespace Optima */
