/*
 * MinimumSolver.hpp
 *
 *  Created on: 16 Nov 2012
 *      Author: Allan
 */

#pragma once

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

// GeoMath includes
#include "Common.hpp"
#include "Scaling.hpp"

namespace Optima {

class MinimumSolver
{
public:
	// Forward declarations
	struct Lagrange;
	struct Options;
	struct Result;

    /**
     * Constructs a default @ref MinimumSolver instance
     */
    MinimumSolver();

    /**
     * Sets the dimension of the minimisation problem
     *
     * @param num_variables The number of variables in the problem
     * @param num_constraints The number of constraints in the problem
     */
    void SetDimension(unsigned num_variables, unsigned num_constraints);

    /**
     * Sets the objective function of the minimisation problem
     *
     * @param objective The objective function defined as @ref Objective
     */
    void SetObjective(const Objective& objective);

    /**
     * Sets the constraint function of the minimisation problem
     *
     * @param constraint The constraint function defined as @ref Constraint
     */
    void SetConstraint(const Constraint& constraint);

    /**
     * Sets the scaling options of the minimisation problem
     *
     * @param scaling The scaling options of the solver
     */
    void SetScaling(const Scaling& scaling);

    /**
     * Sets the general options of the minimisation solver
     *
     * @param options The general options of the solver
     */
    void SetOptions(const Options& options);

    /**
     * Solves the minimisation problem with @c x as initial guess
     *
     * @param x The initial guess of the problem
     * @return An instance of @ref Result
     */
    Result Solve(VectorXd& x);

    /**
     * Solves the minimisation problem with @c x as initial guess and @c lagrange as the initial guess of the Lagrange multipliers
     *
     * @param lagrange The initial guess of the Lagrange multipliers
     * @return An instance of @ref Result
     */
    Result Solve(VectorXd& x, Lagrange& lagrange);

public:
    struct Lagrange
    {
        /// The state of the Lagrangian multipliers @c y with respect to the equality constraints
        VectorXd y;

        /// The state of the Lagrangian multipliers @c z with respect to the nonnegative constraints
        VectorXd z;

        /// Constructs a default @ref Lagrange instance
        Lagrange();

        /// Constructs a @ref Lagrange instance with given Lagrange multipliers
        Lagrange(const VectorXd& y, const VectorXd& z);
    };

    struct Options
    {
        /// The logical flag that indicates if output of the calculation will be performed (default: false)
        bool output;

        /// The maximum number of iterations (default: 100)
        unsigned max_iter;

        /// The tolerance for the optimality error (default: 1.0e-6)
        double residual_tol;

        /// The initial value of the barrier parameter \f$\mu\f$ (default: 0.1)
        double mu;

        /// Constructs a default instance of @ref Options
        Options();
    };

    struct Result
    {
        /// The logical flag that indicates if the calculation converged
        bool converged;

        /// The number of iterations took to converge
        unsigned num_iter;

        /// Constructs a default @ref Result instance
        Result();

        /// Constructs a @ref Result instance with given data
        Result(bool converged, unsigned num_iter);

        /// Converts the @ref Result instance in a boolean
        operator bool();
    };

private:
    /// Initialises the solver with the data in @c x and @c lagrange
    void Initialise(const VectorXd& x, const Lagrange& lagrange);

    /// Calculates the objective function as well as its gradient and hessian
    void UpdateObjectiveFunction();

    /// Calculates the equality constraint function as well as its gradient and hessian
    void UpdateConstraintFunction();

    /// Calculates the penalty-barrier function as well as its gradient
    void UpdateBarrierFunction();

    /// Calculates the extended hessian matrix @c H used in the linear system
    void UpdateExtendedHessian();

    /// Assembles the Newton linear system from which @c dx and @c dy can be calculated
    void UpdateLinearSystem();

    /// Calculates the Newton directions @c dx, @c dy and @c dz by solving the Newton linear system
    void UpdateSearchDirections();

    /// Apply the inertia correction to the linear system in order to have a descent Newton direction
    void ApplyInertiaCorrection();

    /// Apply the adequacy correction to the Newton direction @c dx
    void ApplyAdequacyCorrection();

    /// Calculates the iterates @c x, @c y and @c z by using a line-search algorithm
    void CalculateIterates();

    /// Iterates the state of the solver, calling most of the previous methods and determining the new iterates @c x, @c y and @c z
    void Iterate();

    /// Checks convergence of the calculations
    bool CheckConvergence();

    /// Checks convergence of the subproblem represented by (\f$\mu\f$, \f$\rho\f$)
    bool CheckSubproblemOptimality();

    /// Updates the penalty and barrier parameters \f$\mu\f$ and \f$\rho\f$ respectively
    void UpdatePenaltyBarrierParameters();

    /// Checks if the linear system is ill-conditioned
    bool IsIllConditioned() const;

    /// Checks if the Newton direction @c dx is a descent direction
    bool IsDescentDirection() const;

    /// Calculates the fraction-to-the-boundary constant in accordance with equation (15) Wachter and Biegler (2005)
    double CalculateBoundaryFraction(const VectorXd& p, const VectorXd& dp) const;

    /// Calculates the penalty-barrier function in accordance with equation (24) of Andreani et al. (2012)
    double CalculateBarrierFunction(const VectorXd& x) const;

    /// Calculates the step length via a backtracking line search algorithm in accordance with Andreani et al. (2012)
    double CalculateBacktrackingStepLength() const;

    /// Outputs the header of the calculation output
    void OutputHeader() const;

    /// Outputs the current state of the calculation
    void OutputState(unsigned iter) const;



    double UpdateOptimalityError();

    void UpdateBarrierParameter();

private:
    //==================================================================================================//
    // Miscellaneous data
    //==================================================================================================//

    /// The objective function of the nonlinear optimisation problem
    Objective objective;

    /// The constraint vector function of the nonlinear optimisation problem
    Constraint constraint;

    /// The general options of the nonlinear optimisation problem
    Options options;

    /// The scaling options of the nonlinear optimisation problem
    Scaling scaling;

    /// The dimension of the nonlinear optimisation problem
    unsigned dimx;

    /// The dimension of the nonlinear equality constraints
    unsigned dimy;

    /// The total dimension as the sum of @c dimx and @c dimy
    unsigned dim;

    /// The current state of the iterate @c x
    VectorXd x;

    /// The current state of the Lagrangian multipliers @c y with respect to the equality constraints
    VectorXd y;

    /// The current state of the Lagrangian multipliers @c z with respect to the nonnegative constraints
    VectorXd z;

    /// The current state of the extended Hessian matrix
    MatrixXd H;

    /// The current state of the left-hand side coefficient matrix of the linear system
    MatrixXd lhs;

    /// The current state of the right-hand side vector of the linear system
    VectorXd rhs;

    /// The current state of the solution of the linear system
    VectorXd solution;

    /// The current state of the barrier parameter
    double mu;

    /// The current state of the Newton search direction with respect to @c x
    VectorXd dx;

    /// The current state of the Newton search direction with respect to @c y
    VectorXd dy;

    /// The current state of the Newton search direction with respect to @c z
    VectorXd dz;

    /// The current state of the inertia correction parameter \f$ \lambda \f$
    double delta;

    /// The current state of the KKT residual
    VectorXd residual;

    //==================================================================================================//
    // Objective function data
    //==================================================================================================//

    /// The current state of the objective function at @c x
    double f;

    /// The current state of the gradient of the objective function at @c x
    VectorXd grad_f;

    /// The current state of the hessian of the objective function at @c x
    MatrixXd hessian_f;

    //==================================================================================================//
    // Equality constraint data
    //==================================================================================================//

    /// The current state of the equality constraints at @c x
    VectorXd h;

    /// The current state of the gradient of the equality constraints at @c x
    MatrixXd grad_h;

    /// The current state of the hessian of the equality constraints at @c x
    std::vector<MatrixXd> hessian_h;

    //==================================================================================================//
    // Barrier function data
    //==================================================================================================//

    /// The current state of the barrier function @c x
    double phi;

    /// The current state of the gradient of the barrier function @c x
    VectorXd grad_phi;

    /// The current backtracking step-length
    double alpha_max;

    /// The current boundary fraction
    double alpha_z;
};

} /* namespace Optima */
