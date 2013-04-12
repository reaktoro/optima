/*
 * Experimental.hpp
 *
 *  Created on: 27 Mar 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <vector>
#include <list>

// Eigen includes
#include <Eigen/Core>
#include <Eigen/LU>
using namespace Eigen;

// Optima includes
#include "Filter.hpp"
#include "Common.hpp"

namespace Optima {

const double tau_min = 0.99;

struct OptimumState
{
    VectorXd x, y, z;

    VectorXd dx, dy, dz;

    double alphax, alphay, alphaz;

    ObjectiveState f;

    ConstraintState h;
};

typedef std::function<bool(const OptimumProblem&, OptimumState&)>
    SearchUpdater;

typedef std::function<bool(const OptimumProblem&, const OptimumState&, OptimumState&)>
    StepUpdater;

typedef std::function<bool(const OptimumProblem&, OptimumState&)>
    RestorationUpdater;

class InteriorBarrierNewtonSearchUpdater
{
public:
    InteriorBarrierNewtonSearchUpdater();

    bool operator()(const OptimumProblem& problem, OptimumState& state)
    {
        UpdateBarrierParameter(state);
        UpdateBarrierFunction(state);
        UpdateExtendedHessian(state);
        UpdateLinearSystem(state);
        Update(problem, state);

        return true;
    }

    void UpdateBarrierParameter(const OptimumState& state)
    {
        // This update was taken from equation (3.6) in Nocedal et al. (2009)
        const double xdotz = state.x.dot(state.z);
        const double minxz = state.x.cwiseProduct(state.z).minCoeff();
        const double xi = minxz/xdotz * dim;
        const double aux = std::min(0.05*(1 - xi)/xi, 2.0);
        const double sigma = 0.1*std::pow(aux, 3);

        mu = sigma*xdotz/dim;
    }

    void UpdateBarrierFunction(const OptimumState& state)
    {
        phi.func = state.f.func - mu * std::log(state.x.prod());

        phi.grad.noalias() = state.f.grad;
        phi.grad.noalias() -= mu * state.x.cwiseInverse();
    }

    void UpdateExtendedHessian(const OptimumState& state)
    {
        H = state.f.hessian;

        for(unsigned j = 0; j < state.h.hessian.size(); ++j)
            H.noalias() += state.y[j] * state.h.hessian[j];

        H.diagonal() += state.z.cwiseQuotient(state.x);
    }

    void UpdateLinearSystem(const OptimumState& state)
    {
        rhs.resize(dim);
        lhs.resize(dim, dim);

        lhs.block(0000, 0000, dimx, dimx) = H;
        lhs.block(0000, dimx, dimx, dimy) = state.h.grad.transpose();
        lhs.block(dimx, 0000, dimy, dimx) = state.h.grad;
        lhs.block(dimx, dimx, dimy, dimy) = MatrixXd::Zero(dimy, dimy);

        rhs.segment(0000, dimx) = phi.grad + state.h.grad.transpose() * state.y;
        rhs.segment(dimx, dimy) = state.h.func;
    }

    void Update(const OptimumProblem& problem, OptimumState& state)
    {
        // Solve the reduced KKT linear system
        u = lhs.lu().solve(rhs);

        // Update the Newton search directions dx and dy from u
        state.dx = u.segment(0000, dimx);
        state.dy = u.segment(dimx, dimy);

        // Update the Newton search direction dz
        state.dz = (mu - state.x.array()*state.z.array() -
            state.z.array()*state.dx.array())/state.x.array();

        // Update the fraction to the boundary steps
        state.alphax = CalculateBoundaryFraction(state.x, state.dx);
        state.alphaz = CalculateBoundaryFraction(state.z, state.dz);
        state.alphay = state.alphax;

        // Update the iterates x, y, z
        state.x += state.alphax * state.dx;
        state.y += state.alphay * state.dy;
        state.z += state.alphaz * state.dz;

        // Update the objective and constraint states
        state.f = problem.objective(state.x);
        state.h = problem.constraint(state.x);
    }

    double CalculateBoundaryFraction(const VectorXd& p, const VectorXd& dp) const
    {
        const double tau = std::max(tau_min, 1 - mu);
        double alpha = 1.0;
        for(unsigned i = 0; i < dimx; ++i)
        {
            const double alphai = -tau*p[i]/dp[i];
            if(0 <= alphai and alphai <= 1.0)
                alpha = std::min(alpha, alphai);
        }
        return alpha;
    }

private:
    /// The dimension of the x vector (number of variables)
    unsigned dimx;

    /// The dimension of the y vector (number of constraints)
    unsigned dimy;

    /// The total dimension of the problem
    unsigned dim;

    /// The extended Hessian matrix
    MatrixXd H;

    /// The left-hand side coefficient matrix of the linear system
    MatrixXd lhs;

    /// The right-hand side vector of the linear system
    VectorXd rhs;

    /// The solution of the linear system
    VectorXd u;

    /// The state of the barrier function
    ObjectiveState phi;

    /// The barrier parameter
    double mu;
};

const double gamma_theta = 1.0e-05;
const double gamma_eta   = 1.0e-05;
const double gamma_zeta  = 1.0e-05;
const double gamma_phi   = 1.0e-05;

const double s_theta = 1.1;
const double s_eta   = 1.1;
const double s_zeta  = 1.1;
const double s_phi   = 2.3;

const double eta_theta = 1.0e-04;
const double eta_zeta  = 1.0e-04;
const double eta_phi   = 1.0e-04;

const double xi_alpha = 0.05;

class InteriorFilter4DStepUpdater
{
public:
    InteriorFilter4DStepUpdater(const OptimumState& state)
    {
        UpdateMeasures(state);

        theta_min = 1.0e-04 * std::max(1.0, theta);
        theta_max = 1.0e+04 * std::max(1.0, theta);

        eta_min   = 1.0e-04 * std::max(1.0, eta);
        eta_max   = 1.0e+04 * std::max(1.0, eta);

        zeta_min  = 1.0e-04 * std::max(1.0, zeta);
        zeta_max  = 1.0e+04 * std::max(1.0, zeta);

        filter.Add({theta_max, eta_max, zeta_max, 1.0e+200});
    }

    bool operator()(const OptimumProblem& problem, const OptimumState& full, OptimumState& state)
    {
        return true;
    }

    double CalculateAlphaMin(const OptimumState& state) const
    {
        const double pi1 = -gamma_phi*theta/dfdx;
        const double pi2 = theta_s/dfdx_phi;
        const double pi3 = eta_s/dfdx_phi;
        const double pi4 = zeta_s/dfdx_phi;

        if(theta <= theta_min and eta <= eta_min and zeta <= zeta_min)
            return std::min({gamma_theta, pi1, pi2, pi3, pi4});
    }

    void UpdateMeasures(const OptimumState& state)
    {
        theta = state.h.func.norm();
        eta   = (state.f.grad - state.h.grad.transpose()*state.y - state.z).norm();
        zeta  = (mu*state.x.cwiseInverse() - state.z).norm();
        phi   = state.f.func;
    }

    void Update(const OptimumState& state)
    {
        if(dfdx < 0.0)
        {
            theta_s  = std::pow(theta, s_theta);
            eta_s    = std::pow(eta,   s_eta);
            zeta_s   = std::pow(zeta,  s_zeta);
            dfdx_phi = std::pow(-dfdx, s_phi);

            const double pi1 = -gamma_phi*theta/dfdx;
            const double pi2 = theta_s/dfdx_phi;
            const double pi3 = eta_s/dfdx_phi;
            const double pi4 = zeta_s/dfdx_phi;

            alpha_min = (theta <= theta_min and eta <= eta_min and zeta <= zeta_min) ?
                std::min({gamma_theta, pi1, pi2, pi3, pi4}) : std::min({gamma_theta, pi1});

            const double alphax_max = -state.x.cwiseQuotient(state.dx).cwiseMin(VectorXd::Zero(dimx)).maxCoeff();
            const double alphaz_max = -state.z.cwiseQuotient(state.dz).cwiseMin(VectorXd::Zero(dimx)).maxCoeff();

            alpha_max = std::min(1.0, 0.95 * std::min(alphax_max, alphaz_max));
        }
    }

private:
    unsigned dimx;

    unsigned dimy;

    unsigned dim;

    double dfdx;

    double theta;

    double eta;

    double zeta;

    double phi;

    double mu;

    Filter filter;

    double theta_min;

    double eta_min;

    double zeta_min;


    double theta_max;

    double eta_max;

    double zeta_max;


    double alpha_min;

    double alpha_max;


    double dfdx_phi;

    double theta_s;

    double eta_s;

    double zeta_s;
};

} /* namespace Optima */
