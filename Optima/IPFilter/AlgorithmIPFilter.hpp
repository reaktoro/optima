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

    double CalculateLargestDelta(const State& state) const;

    double CalculateBacktrackingDelta(const State& state) const;

public:
    struct Options
    {
        OptimalityMeasure psi;

        unsigned max_iter;

        double tolerance;
    };

    /**
     * The state of the algorithm data
     */
    struct State
    {
        State(const OptimumProblem& problem, const Params& params, const Options& options);

        void Initialise(const VectorXd& x, const VectorXd& y, const VectorXd& z);

        /// Updates all quantities that are dependent on delta
        void UpdateDelta(double delta);

        void AcceptTrialPoint();

        /// Calculates the largest trust-region radius that satisfies the positivity conditions
        double CalculateLargestDelta() const;

        /// Searches for a trust-region radius that satisfies the centrality neighborhood conditions
        void SearchDeltaNeighborhood() throw(SearchDeltaNeighborhoodError);

        void SearchDelta() throw(SearchDeltaError);

        void SearchDeltaRestoration() throw(SearchDeltaRestorationError);

        void ComputeSteps();

        void SolveRestoration() throw(MaxIterationError);

        void Solve() throw(MaxIterationError);

        double CalculatePsi() const;

        double CalculateLinearModel() const;

        bool PassStoppingCriteria() const;

        bool PassRestorationCondition() const;

        bool PassFilterCondition() const;

        unsigned dimx, dimy;

        const OptimumProblem& problem;

        const Params& params;

        const Options& options;

        VectorXd snx, stx;

        VectorXd sny, sty;

        VectorXd snz, stz;

        double norm_sn, norm_st;

        Filter filter;

        double alpha_n, alpha_t;

        double delta;

        double delta_max;

        VectorXd x, y, z;

        VectorXd x_old, y_old, z_old;

        ObjectiveState f, f_old;

        ConstraintState h, h_old;

        double mu, mu_old;

        double thh, thc, thl;

        double thh_old, thc_old, thl_old;

        double theta, theta_old;

        double psi, psi_old;

        double m, m_old;

        double c;

        double sigma;

        double gamma;

        double M;

        unsigned iter;

        PartialPivLU<MatrixXd> lu;

        MatrixXd lhs;

        VectorXd rhs, u;

        MatrixXd H;
    };

private:
    OptimumProblem problem;
};

} /* namespace IPFilter */
} /* namespace Optima */
