/*
 * IPFilterSolver.hpp
 *
 *  Created on: 4 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <set>

// Eigen includes
#include <Eigen/Dense>
using namespace Eigen;

// Optima includes
#include <Optima/IPFilter/IPFilterErrors.hpp>
#include <Optima/IPFilter/IPFilterOptions.hpp>
#include <Optima/IPFilter/IPFilterParams.hpp>
#include <Optima/IPFilter/IPFilterResult.hpp>
#include <Optima/IPFilter/IPFilterState.hpp>
#include <Optima/Misc/QualitySolver.hpp>
#include <Optima/Utils/Filter.hpp>
#include <Optima/Utils/OptimumProblem.hpp>
#include <Optima/Utils/Outputter.hpp>
#include <Optima/Utils/Scaling.hpp>

namespace Optima {

/**
 * The primal-dual interior-point optimisation solver based on the ipfilter algorithm
 */
class IPFilterSolver
{
public:
    /**
     * Constructs a default @ref IPFilterSolver instance
     */
    IPFilterSolver();

    /**
     * Sets the options for the optimisation calculation
     */
    void SetOptions(const IPFilterOptions& options);

    /**
     * Sets the parameters of the optimisation algorithm
     */
    void SetParams(const IPFilterParams& params);

    /**
     * Sets the definition of the optimisation problem
     */
    void SetProblem(const OptimumProblem& problem);

    /**
     * Sets the scaling factors for the optimisation problem
     * @param scaling
     */
    void SetScaling(const Scaling& scaling);

    /**
     * Gets the calculation options of the optimisation solver
     */
    const IPFilterOptions& GetOptions() const;

    /**
     * Gets the algorithm params of the optimisation solver
     */
    const IPFilterParams& GetParams() const;

    /**
     * Gets the solution result of the last optimisation calculation
     */
    const IPFilterResult& GetResult() const;

    /**
     * Gets the solution state of the last optimisation calculation
     */
    const IPFilterState& GetState() const;

    /**
     * Gets the optimisation problem of the optimisation solver
     */
    const OptimumProblem& GetProblem() const;

    /**
     * Checks if the optimisation calculation has already converged
     */
    bool Converged() const;

    /**
     * Initialises the internal state of the optimisation solver
     *
     * This method should be called prior to any call to method @ref Iterate. It will set the initial guesses of
     * iterates @b x, @b y, and @b z and compute internal parameters that depend on these data.
     *
     * The method scale the inputs @c x, @c y, and @c z according to the scaling specified in @ref SetScaling.
     *
     * @param[in,out] x The initial guess of iterate @b x as input. The scaled initial guess as output.
     * @param[in,out] y The initial guess of iterate @b y as input. The scaled initial guess as output.
     * @param[in,out] z The initial guess of iterate @b z as input. The scaled initial guess as output.
     */
    void Initialise(VectorXd& x, VectorXd& y, VectorXd& z);

    /**
     * Performs a single iteration towards the solution of the optimisation problem
     *
     * This method allows the user to provide the initial guess for the primal variables @b x as well as the Lagrange
     * multipliers @b y and @b z.
     *
     * @param[in,out] x The current state of iterate @b x as input. Its improved state at the end of the iteration as output.
     * @param[in,out] y The current state of iterate @b y as input. Its improved state at the end of the iteration as output.
     * @param[in,out] z The current state of iterate @b z as input. Its improved state at the end of the iteration as output.
     */
    void Iterate(VectorXd& x, VectorXd& y, VectorXd& z);

    /**
     * Solves the optimisation problem
     *
     * This method allows the user to provide only the initial guess for the primal variables @b x.
     *
     * The initial guess of the Lagrange multipliers @b y and @b z are given by the options
     * @ref IPFilterOptions::InitialGuessOptions::y and @ref IPFilterOptions::InitialGuessOptions::z in
     * IPFilterOptions::initialguess.
     *
     * @param[in,out] x The initial guess of the primal variables as input. The optimum solution at the end of
     *     the calculation as output.
     */
    void Solve(VectorXd& x);

    /**
     * Solves the optimisation problem
     *
     * This method allows the user to provide the initial guess for the primal variables @b x as well as the Lagrange
     * multipliers @b y and @b z.
     *
     * This is usefull for sequential calculations where the i-th calculation uses the result of the (i-1)-th
     * calculation as initial guess. Therefore, convergence to an optimal point might is achieved in less iterations.
     *
     * Note, however, that some components of parameters @c x and @c z might be modified in order to improve
     * robustness and efficiency. The modification is given by:
     *
     *  - <tt> x = max(x, xguessmin) </tt> (see @ref IPFilterOptions::InitialGuessOptions::xmin),
     *  - <tt> z = max(z, zguessmin) </tt> (see @ref IPFilterOptions::InitialGuessOptions::zmin).
     *
     * @param[in,out] x The initial guess of iterate @b x as input. The optimum solution at the end of the calculation as output.
     * @param[in,out] y The initial guess of iterate @b y as input. The optimum solution at the end of the calculation as output.
     * @param[in,out] z The initial guess of iterate @b z as input. The optimum solution at the end of the calculation as output.
     */
    void Solve(VectorXd& x, VectorXd& y, VectorXd& z);

private:
    /// The definition of the optimisation problem
    OptimumProblem problem;

    /// The parameters used for the optimisation calculation
    IPFilterParams params;

    /// The options used for the optimisation calculation
    IPFilterOptions options;

    /// The result details of the last optimisation calculation
    IPFilterResult result;

    /// The scaling factors for the optimisation problem
    Scaling scaling;

    /// The output instance for printing the calculation progress
    Outputter outputter;

    /// The filter used during the search for a suitable trust-region radius
    Filter filter;

    /// The quality solver used to calculate the sigma parameter based on the quality function approach
    QualitySolver quality;

    /// The dimension of the objective and constraint functions
    unsigned dimx, dimy;

    /// The previous, current and next solution states
    IPFilterState prev, curr, next;

    /// The x-components of the normal and tangencial steps
    VectorXd snx, stx;

    /// The y-components of the normal and tangencial steps
    VectorXd sny, sty;

    /// The z-components of the normal and tangencial steps
    VectorXd snz, stz;

    /// The x-, y- and z-components of the Newton step vector
    VectorXd dx, dy, dz;

    /// The perturbation parameter for the Newton iterations
    double mu;

    /// The norms of the normal and tangencial steps respectively
    double norm_sn, norm_st;

    /// The step lengths of the normal and tangencial trust-region step vectors
    double alphan, alphat;

    /// The step lengths of the x- and z-components of the Newton step vector
    double alphax, alphaz;

    /// The current radius of the trust-region
    double delta;

    /// The initial value of the trust-region radius used for the trust-region search
    double delta_initial;

    /// The parameter c used for the calculation of the psi measure
    double c;

    /// The parameter gamma used in the neighborhood condition checking
    double gamma;

    /// The parameter M used in the neighborhood condition checking
    double M;

    /// The LU decomposition of the reduced KKT matrix
    PartialPivLU<MatrixXd> lu;

    /// The reduced KKT matrix
    MatrixXd lhs;

    /// The right-hand side vector of the linear system and the auxiliary linear system solution
    VectorXd rhs, u;

    /// The gradient of the Lagrange function with respect to x at the current state
    VectorXd Lx;

    /// The Hessian of the Lagrange function with respect to x at the current state
    MatrixXd Lxx;

    /// The boolean flag that indicates if the watchdog strategy is currently in use
    bool watchdog;

private:
    bool AnyFloatingPointException(const IPFilterState& state) const;
    bool PassFilterCondition() const;
    bool PassRestorationCondition(double delta) const;
    bool PassSafeStepCondition() const;

    double CalculateDeltaPositiveXZ() const;
    double CalculateDeltaXzGreaterGammaMu() const;
    double CalculateLargestBoundaryStep(const VectorXd& p, const VectorXd& dp) const;
    double CalculateLargestQuadraticStep(const VectorXd& a, const VectorXd& b, const VectorXd& c, const VectorXd& d) const;
    double CalculateNextLinearModel() const;
    double CalculatePsi(const IPFilterState& state) const;
    double CalculateSigma() const;
    double CalculateSigmaDefault() const;
    double CalculateSigmaLOQO() const;
    double CalculateSigmaQuality();
    double CalculateSigmaRestoration() const;
    double CalculateSigmaSafeStep() const;

    void AcceptTrialPoint();
    void ExtendFilter();
    void InitialiseAuxiliary(VectorXd& x, VectorXd& y, VectorXd& z);
    void InitialiseOutputter();
    void IterateNewton(VectorXd& x, VectorXd& y, VectorXd& z);
    void IterateTrustRegion(VectorXd& x, VectorXd& y, VectorXd& z);
    void OutputHeader();
    void OutputState();
    void SearchDeltaNeighborhood();
    void SearchDeltaTrustRegion();
    void SearchDeltaTrustRegionRestoration();
    void SolveRestoration();
    void UpdateNewtonNextState();
    void UpdateNewtonSteps();
    void UpdateState(const VectorXd& x, const VectorXd& y, const VectorXd& z, IPFilterState& state);
    void UpdateTrustRegionNextState(double delta);
    void UpdateTrustRegionSafeTangentialStep();
    void UpdateTrustRegionSteps();
    void UpdateTrustRegionStepsRestoration();
};

} /* namespace Optima */
