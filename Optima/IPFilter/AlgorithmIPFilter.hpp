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

    void SetParams(const Params& params);

    void SetOptions(const Options& options);

    void SetProblem(const OptimumProblem& problem);

    void Initialise(const State& state);

    void Solve() throw(MaxIterationError);

private:

    void SetState(const VectorXd& x, const VectorXd& y, const VectorXd& z, State& state) const;

    void UpdateNextState(double delta);

    void AcceptTrialPoint();

    /// Calculates the largest trust-region radius that satisfies the positivity conditions
    double CalculateLargestDelta() const;

    /// Searches for a trust-region radius that satisfies the centrality neighborhood conditions
    void SearchDeltaNeighborhood() throw(SearchDeltaNeighborhoodError);

    void SearchDelta() throw(SearchDeltaError);

    void SearchDeltaRestoration() throw(SearchDeltaRestorationError);

    void ComputeSteps();

    void SolveRestoration() throw(MaxIterationError);

    double CalculateNextLinearModel() const;

    double CalculateSigma() const;

    bool PassStoppingCriteria() const;

    bool PassRestorationCondition() const;

    bool PassFilterCondition() const;

public:
    struct Options
    {
        unsigned psi_scheme;

        unsigned max_iter;

        double tolerance;
    };

    /**
     * The state of the algorithm data
     */
    struct State
    {
        State();

        VectorXd x, y, z;

        ObjectiveState f;

        ConstraintState h;

        double mu;

        double thh, thc, thl;

        double theta;

        double psi;
    };

private:
    OptimumProblem problem;

    Params params;

    Options options;

    unsigned dimx, dimy;

    State curr, next;

    VectorXd snx, stx;

    VectorXd sny, sty;

    VectorXd snz, stz;

    double norm_sn, norm_st;

    double alphan, alphat;

    Filter filter;

    double delta;

    double delta_max;

    double c;

    double gamma;

    double M;

    unsigned iter;

    bool restoration;

    bool safe_step;

    PartialPivLU<MatrixXd> lu;

    MatrixXd lhs;

    VectorXd rhs, u;

    MatrixXd H;
};

} /* namespace IPFilter */
} /* namespace Optima */
