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
#include <IPFilter/IPFilterErrors.hpp>
#include <IPFilter/IPFilterOptions.hpp>
#include <IPFilter/IPFilterParams.hpp>
#include <IPFilter/IPFilterResult.hpp>
#include <IPFilter/IPFilterState.hpp>
#include <Utils/ActiveMonitoring.hpp>
#include <Utils/Filter.hpp>
#include <Utils/OptimumProblem.hpp>
#include <Utils/Outputter.hpp>
#include <Utils/Scaling.hpp>

namespace Optima {

/**
 * The primal-dial interior-point non-convex optimisation solver based on the ipfilter algorithm
 */
class IPFilterSolver
{
public:
    typedef IPFilter::Options Options;
    typedef IPFilter::Params  Params;
    typedef IPFilter::Result  Result;
    typedef IPFilter::State   State;

    // The possible errors that might happen with the IPFilter algorithm
    typedef IPFilter::ErrorInitialGuess                      ErrorInitialGuess;
    typedef IPFilter::ErrorInitialGuessActivePartition       ErrorInitialGuessActivePartition;
    typedef IPFilter::ErrorInitialGuessFloatingPoint         ErrorInitialGuessFloatingPoint;
    typedef IPFilter::ErrorIteration                         ErrorIteration;
    typedef IPFilter::ErrorIterationMaximumLimit             ErrorIterationMaximumLimit;
    typedef IPFilter::ErrorSearchDelta                       ErrorSearchDelta;
    typedef IPFilter::ErrorSearchDeltaNeighborhood           ErrorSearchDeltaNeighborhood;
    typedef IPFilter::ErrorSearchDeltaTrustRegion            ErrorSearchDeltaTrustRegion;
    typedef IPFilter::ErrorSearchDeltaTrustRegionRestoration ErrorSearchDeltaTrustRegionRestoration;

    /**
     * Constructs a default @ref IPFilterSolver instance
     */
    IPFilterSolver();

    /**
     * Sets the options for the optimisation calculation
     */
    void SetOptions(const Options& options);

    /**
     * Sets the parameters of the optimisation algorithm
     */
    void SetParams(const Params& params);

    /**
     * Sets the definition of the optimisation problem
     */
    void SetProblem(const OptimumProblem& problem);

    /**
     * Sets the scaling factors for the optimisation problem
     * @param scaling
     */
    void SetScaling(const Scaling& scaling);

    void SetActiveMonitoring(const ActiveMonitoring& active_monitor);

    const ActiveMonitoring& GetActiveMonitoring() const;

    /**
     * Gets the calculation options of the optimisation solver
     */
    const Options& GetOptions() const;

    /**
     * Gets the algorithm params of the optimisation solver
     */
    const Params& GetParams() const;

    /**
     * Gets the solution result of the last optimisation calculation
     */
    const Result& GetResult() const;

    /**
     * Gets the solution state of the last optimisation calculation
     */
    const State& GetState() const;

    /**
     * Gets the optimisation problem of the optimisation solver
     */
    const OptimumProblem& GetProblem() const;

    /**
     * Solves the optimisation problem
     *
     * This method allows the user to provide only the initial guess
     * for the primal variables @a x. The initial guess of the Lagrange
     * multipliers @a y and @a z are automatically estimated.
     *
     * @param x The initial guess of the primal variables @a x
     */
    void Solve(VectorXd& x);

    /**
     * Solves the optimisation problem
     *
     * This method allows the user to provide the initial guess
     * for the primal variables @a x as well as the Lagrange
     * multipliers @a y and @a z.
     *
     * This is usefull for sequential calculations where the
     * i-th calculation uses the result of the (i-1)-th
     * calculation as initial guess. Therefore, convergence
     * to an optimal point might result in less iterations.
     *
     * @param x The initial guess of the primal variables @a x
     * @param y The initial guess of the Lagrange multipliers @a y
     * @param z The initial guess of the Lagrange multipliers @a z
     */
    void Solve(VectorXd& x, VectorXd& y, VectorXd& z);

private:
    bool AnyDepartingActivePartition() const;
    bool AnyFloatingPointException(const State& state) const;
    bool PassConvergenceCondition() const;
    bool PassFilterCondition() const;
    bool PassRestorationCondition(double delta) const;
    bool PassSafeStepCondition() const;

    double CalculateDeltaPositiveXZ() const;
    double CalculateDeltaXzGreaterGammaMu() const;
    double CalculateLargestBoundaryStep(const VectorXd& p, const VectorXd& dp) const;
    double CalculateLargestQuadraticStep(const VectorXd& a, const VectorXd& b, const VectorXd& c, const VectorXd& d) const;
    double CalculateNextLinearModel() const;
    double CalculatePsi(const State& state) const;
    double CalculateSigma() const;
    double CalculateSigmaDefault() const;
    double CalculateSigmaLOQO() const;

    void AcceptTrialPoint();
    void ExtendFilter();
    void Initialise(VectorXd& x, VectorXd& y, VectorXd& z);
    void OutputHeader();
    void OutputState();
    void ResetLagrangeMultipliersZ(State& state) const;
    void SearchDeltaNeighborhood();
    void SearchDeltaTrustRegion();
    void SearchDeltaTrustRegionRestoration();
    void Solve();
    void SolveRestoration();
    void UpdateActiveMonitor();
    void UpdateNeighborhoodParameterM();
    void UpdateNextState(double delta);
    void UpdateNormalTangentialSteps();
    void UpdateNormalTangentialStepsRestoration();
    void UpdateSafeTangentialStep();
    void UpdateState(const VectorXd& x, const VectorXd& y, const VectorXd& z, State& state);

private:
    /// The definition of the optimisation problem
    OptimumProblem problem;

    /// The parameters used for the optimisation calculation
    Params params;

    /// The options used for the optimisation calculation
    Options options;

    /// The result details of the last optimisation calculation
    Result result;

    /// The scaling factors for the optimisation problem
    Scaling scaling;

    /// The partitioning of the primal variables that are active at the bounds simultaneously
    ActiveMonitoring active_monitor;

    /// The output instance for printing the calculation progress
    Outputter outputter;

    /// The dimension of the objective and constraint functions respectively
    unsigned dimx, dimy;

    /// The current and next states respectively
    State curr, next;

    /// The x-component of the normal and tangencial steps respectively
    VectorXd snx, stx;

    /// The y-component of the normal and tangencial steps respectively
    VectorXd sny, sty;

    /// The z-component of the normal and tangencial steps respectively
    VectorXd snz, stz;

    /// The norms of the normal and tangencial steps respectively
    double norm_sn, norm_st;

    /// The normal and tangencial step-lengths respectively
    double alphan, alphat;

    /// The filter used during the search for a suitable trust-region radius
    Filter filter;

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

    /// The flag that indicates if the algorithm is currently in the restoration mode
    bool restoration;

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
};

} /* namespace Optima */
