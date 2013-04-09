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
    };

    /**
     * The state of the algorithm data
     */
    struct State
    {
        /// Updates all quantities that are dependent on delta
        void UpdateDelta(double delta);

        /// Calculates the largest trust-region radius that satisfies the positivity conditions
        double CalculateLargestDelta() const;

        /// Searches for a trust-region radius that satisfies the centrality neighborhood conditions
        void SearchDeltaNeighborhood();

        void SearchDeltaTrialTests();

        void SearchDeltaRestoration();

        void UpdateNormalTangentialSteps();

        double CalculatePsi() const;

        double CalculateLinearModel() const;

        double CalculateRhoRestoration() const;

        unsigned dimx, dimy;

        const OptimumProblem& problem;

        const Params& params;

        const Options& options;

        VectorXd x, y, z;

        VectorXd x_old, y_old, z_old;

        VectorXd snx, stx;

        VectorXd sny, sty;

        VectorXd snz, stz;

        ObjectiveState f, f_old;

        ConstraintState h, h_old;

        Filter filter;

        double norm_sn, norm_st;

        double alpha_n, alpha_t;

        double delta;

        double delta_max;

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

        PartialPivLU<MatrixXd> lu;

        MatrixXd lhs;

        VectorXd rhs_n, rhs_t;

        VectorXd un, ut;

        MatrixXd H;
    };

private:
    OptimumProblem problem;
};

} /* namespace IPFilter */
} /* namespace Optima */
