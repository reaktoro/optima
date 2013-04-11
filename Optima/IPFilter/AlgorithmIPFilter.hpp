/*
 * AlgorithmIPFilter.hpp
 *
 *  Created on: 4 Apr 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

// Optima includes
#include "Common.hpp"
#include "Exceptions.hpp"
#include "Filter.hpp"
#include "Output.hpp"
#include "Utils.hpp"
#include <IPFilter/Params.hpp>

namespace Optima {
namespace IPFilter {

class AlgorithmIPFilter
{
public:
    struct State;
    struct Options;

    AlgorithmIPFilter();

    void SetOptions(const Options& options);

    void SetParams(const Params& params);

    void SetProblem(const OptimumProblem& problem);

    void Solve(State& state);

    void Solve(VectorXd& x);

private:

    bool AnyFloatingPointException(const State& state) const;
    bool PassFilterCondition() const;
    bool PassRestorationCondition() const;
    bool PassStoppingCriteria() const;

    double CalculateDeltaPositivity() const;
    double CalculateNextLinearModel() const;
    double CalculateSigma() const;

    void AcceptTrialPoint();
    void ExtendFilter();
    void Initialise(const State& state);
    void Initialise(const VectorXd& x);
    void OutputHeader() const;
    void OutputState() const;
    void ResetLagrangeMultipliersZ(State& state) const;
    void SearchDeltaNeighborhood();
    void SearchDeltaTrustRegion();
    void SearchDeltaTrustRegionRestoration();
    void Solve();
    void SolveRestoration();
    void UpdateNeighborhoodParameterM();
    void UpdateNextState(double delta);
    void UpdateNormalTangentialSteps();
    void UpdateState(const VectorXd& x, const VectorXd& y, const VectorXd& z, State& state) const;

public:
    /**
     * The options used for the algorithm
     */
    struct Options
    {
        /**
         * The scheme used for the calculation of the psi optimality measure
         */
        unsigned psi_scheme = 0;

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
        bool output = false;
    };

    /**
     * The algorithmic state at the point (x,y,z)
     */
    struct State
    {
        /// Constructs a default @ref State instance
        State();

        /// The iterates x, y, z of the algorithm
        VectorXd x, y, z;

        /// The state of the objective function at x
        ObjectiveState f;

        /// The state of the constraint function at x
        ConstraintState h;

        /// The barrier parameter at (x,z)
        double mu;

        /// The auxiliary theta_h, theta_c, and theta_l optimality measures
        double thh, thc, thl;

        /// The theta optimality measure
        double theta;

        /// The psi optimality measure
        double psi;
    };

private:
    /// The definition of the optimisation problem
    OptimumProblem problem;

    /// The parameters used for the optimisation calculation
    Params params;

    /// The options used for the optimisation calculation
    Options options;

    /// The output instance for printing the calculation progress
    Output output;

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

    /// The current maximum allowed radius of the trust-region
    double delta_max;

    /// The parameter c used for the calculation of the psi measure (initialised at the beginning of the calculation)
    double c;

    /// The parameter gamma used in the neighborhood condition checking (initialised at the beginning of the calculation)
    double gamma;

    /// The parameter M used in the neighborhood condition checking (initialised at the beginning of the calculation)
    double M;

    /// The current iteration number
    unsigned iter;

    /// The flag that indicates if the algorithm is currently in the restoration mode
    bool restoration;

    /// The flag that indicates if the algorithm is currently in the safe tangencial step mode
    bool safe_step;

    /// The LU decomposition of the reduced KKT matrix
    PartialPivLU<MatrixXd> lu;

    /// The reduced KKT matrix
    MatrixXd lhs;

    /// The right-hand side vector of the linear system and the auxiliary linear system solution
    VectorXd rhs, u;

    /// The auxiliary matrix used to assemble the hessian of the Lagrange function
    MatrixXd H;
};

} /* namespace IPFilter */
} /* namespace Optima */
