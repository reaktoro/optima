/*
 * IPFilterOptions.hpp
 *
 *  Created on: 11 Apr 2013
 *      Author: allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// Optima includes
#include <Optima/Utils/Outputter.hpp>

namespace Optima {

/**
 * The options for the optimisation calculation
 */
struct IPFilterOptions
{
    /**
     * The options for the output of the optimisation calculation
     */
    struct OutputOptions : public Outputter::Options
    {
        /**
         * The boolean flag that indicates if the output of the iterates should be scaled
         */
        bool scaled = false;

        /**
         * The boolean flag that indicates if the iteration number should be printed
         */
        bool iter = true;

        /**
         * The boolean flag that indicates if the primal iterate @b x should be printed
         */
        bool x = true;

        /**
         * The boolean flag that indicates if the dual iterate @b y should be printed
         */
        bool y = true;

        /**
         * The boolean flag that indicates if the dual iterate @b z should be printed
         */
        bool z = true;

        /**
         * The boolean flag that indicates if the objective function should be printed
         */
        bool f = true;

        /**
         * The boolean flag that indicates if the norm of the constraint function should be printed
         */
        bool h = true;

        /**
         * The boolean flag that indicates if the perturbation parameter @f$\mu@f$ should be printed
         */
        bool mu = true;

        /**
         * The boolean flag that indicates if the maximum among the feasibility, centrality, and optimality errors should be printed
         */
        bool error = true;

        /**
         * The boolean flag that indicates if the residual error defined as the norm of the KKT equations should be printed
         */
        bool residual = true;

        /**
         * The boolean flag that indicates if the normal step-length @f$\alpha^{n}@f$ should be printed
         */
        bool alphan = true;

        /**
         * The boolean flag that indicates if the tangencial step-length @f$\alpha^{t}@f$ should be printed
         */
        bool alphat = true;

        /**
         * The boolean flag that indicates if the radius of the trust-region @f$\Delta@f$ should be printed
         */
        bool delta = true;

        /**
         * The boolean flag that indicates if the optimality measure @f$\theta@f$ should be printed
         */
        bool theta = true;

        /**
         * The boolean flag that indicates if the optimality measure @f$\psi@f$ should be printed
         */
        bool psi = true;

        /**
         * The boolean flag that indicates if the optimality measure @f$\theta_{c}@f$ should be printed
         */
        bool thc = true;

        /**
         * The boolean flag that indicates if the optimality measure @f$\theta_{h}@f$ should be printed
         */
        bool thh = true;

        /**
         * The boolean flag that indicates if the optimality measure @f$\theta_{l}@f$ should be printed
         */
        bool thl = true;
    };

    /**
     * The options for the initial guess of the optimisation calculation
     */
    struct InitialGuessOptions
    {
        /**
         * The initial guess for the primal variables @b x in the absence of a proper user-provided one
         */
        double x = 1.0;

        /**
         * The initial guess for the dual variables @b y in the absence of a proper user-provided one
         */
        double y = 0.0;

        /**
         * The initial guess for the dual variables @b z in the absence of a proper user-provided one
         */
        double z = 0.01;

        /**
         * The lower bound on the initial guess of the primal variables @b x
         */
        double xmin = 1.0e-16;

        /**
         * The lower bound on the initial guess of the dual variables @b z
         */
        double zmin = 1.0e-16;
    };

    /**
     * The maximum number of iterations allowed in the algorithm
     */
    unsigned max_iterations = 1000;

    /**
     * The tolerance parameter used for the stopping criteria of the algorithm based on optimality
     */
    double tolerance1 = 1.0e-06;

    /**
     * The tolerance parameter used for the stopping criteria of the algorithm based on step size
     */
    double tolerance2 = 1.0e-9;

    /**
     * The options for the output of the optimisation calculation
     */
    OutputOptions output;

    /**
     * The options for the initial guess of the optimisation calculation
     */
    InitialGuessOptions initialguess;
};

} /* namespace Optima */
