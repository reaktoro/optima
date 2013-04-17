/*
 * IPFilterSolver.hpp
 *
 *  Created on: 4 Apr 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Dense>
using namespace Eigen;

// Optima includes
#include <IPFilter/IPFilterOptions.hpp>
#include <IPFilter/IPFilterParams.hpp>
#include <IPFilter/IPFilterResult.hpp>
#include <IPFilter/IPFilterState.hpp>
#include <Utils/Filter.hpp>
#include <Utils/OptimumProblem.hpp>
#include <Utils/Outputter.hpp>

namespace Optima {

/**
 * The primal-dial interior-point non-convex minimisation solver based on the ipfilter algorithm
 */
class IPFilterSolver
{
public:
    typedef IPFilterOptions Options;
    typedef IPFilterParams Params;
    typedef IPFilterResult Result;
    typedef IPFilterState State;

    /**
     * Constructs a default @ref IPFilterSolver instance
     */
    IPFilterSolver();

    /**
     * Sets the options for the minimisation calculation
     */
    void SetOptions(const Options& options);

    /**
     * Sets the parameters of the minimisation algorithm
     */
    void SetParams(const Params& params);

    /**
     * Sets the definition of the minimisation problem
     */
    void SetProblem(const OptimumProblem& problem);

    /**
     * Gets the calculation options of the minimisation solver
     */
    const Options& GetOptions() const;

    /**
     * Gets the algorithm params of the minimisation solver
     */
    const Params& GetParams() const;

    /**
     * Gets the solution result of the last minimisation calculation
     */
    const Result& GetResult() const;

    /**
     * Gets the solution state of the last minimisation calculation
     */
    const State& GetState() const;

    /**
     * Gets the optimisation problem of the minimisation solver
     */
    const OptimumProblem& GetProblem() const;

    /**
     * Solves the minimisation problem
     */
    void Solve(State& state);

    /**
     * Solves the minimisation problem
     */
    void Solve(VectorXd& x);

private:

    bool AnyFloatingPointException(const State& state) const;
    bool PassFilterCondition() const;
    bool PassRestorationCondition() const;
    bool PassSafeStepCondition() const;
    bool PassConvergenceCondition() const;

    double CalculateDeltaPositivity() const;
    double CalculateLargestBoundaryStep(const VectorXd& p, const VectorXd& dp) const;
    double CalculateLargestQuadraticStep(const VectorXd& a, const VectorXd& b, const VectorXd& c, const VectorXd& d) const;
    double CalculateNextLinearModel() const;
    double CalculatePsi(const State& state) const;
    double CalculateSigma() const;
    double CalculateSigmaDefault() const;
    double CalculateSigmaLOQO() const;

    void AcceptTrialPoint();
    void ExtendFilter();
    void Initialise(const State& state);
    void Initialise(const VectorXd& x);
    void OutputHeader();
    void OutputState();
    void ResetLagrangeMultipliersZ(State& state) const;
    void SearchDeltaNeighborhood();
    void SearchDeltaTrustRegion();
    void SearchDeltaTrustRegionRestoration();
    void Solve();
    void SolveRestoration();
    void UpdateNeighborhoodParameterM();
    void UpdateNextState(double delta);
    void UpdateNormalTangentialSteps();
    void UpdateSafeTangentialStep();
    void UpdateState(const VectorXd& x, const VectorXd& y, const VectorXd& z, State& state);

private:
    /// The definition of the optimisation problem
    OptimumProblem problem;

    /// The parameters used for the optimisation calculation
    Params params;

    /// The options used for the optimisation calculation
    Options options;

    /// The result details of the last minimisation calculation
    Result result;

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

    /// The current initial value of the trust-region radius used search a radius that satisfies the neighborhood conditions
    double delta_initial;

    /// The parameter c used for the calculation of the psi measure (initialised at the beginning of the calculation)
    double c;

    /// The parameter gamma used in the neighborhood condition checking (initialised at the beginning of the calculation)
    double gamma;

    /// The parameter M used in the neighborhood condition checking (initialised at the beginning of the calculation)
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
