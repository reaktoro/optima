/*
 * MinimumSolver.cpp
 *
 *  Created on: 16 Nov 2012
 *      Author: Allan
 */

#include "MinimumSolver.hpp"

// C++ includes
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>

// Eigen includes
#include <Eigen/LU>

namespace Optima {
namespace {

const double s_max = 100;

const double kappa_eps = 10;

const double kappa_mu = 0.2;

const double kappa_rho = 5.0;

const double theta_mu = 1.5;

const double theta_rho = 1.5;

const double tau_min = 0.99;

const double gamma_theta = 1.0e-5;

const double gamma_phi = 1.0e-5;

const double gamma_alpha = 0.05;

const double s_theta = 1.1;

const double s_phi = 2.3;

const double eta_phi = 1.0e-4;

const double kappa_soc = 0.99;

const double p_max = 4.0;

const double alpha_eps = 1.0e-16;

const double singular_eps = 1.0e-8;

const double delta_start = 1.0e-4;

const double delta_min = 1.0e-20;

const double delta_max = 1.0e+40;

const double kappa_plus = 8;

const double kappa_plus_plus = 100;

const double kappa_minus = 0.333333;

const double theta = 1.0e-6;

const double beta_min = 1.0e-4;

const double beta_max = 1.0e+4;

//-------------------------------------------------------------------------------------------------
const double gamma_theta_f = 1.0e-05;
const double gamma_theta_o = 1.0e-05;
const double gamma_theta_c = 1.0e-05;

const double delta = 0.1;

const double xi = 0.05;

const double s_f = 1.1;
const double s_o = 2.3;

const double eta = 1.0e-04;

const double eta_theta_f = 1.0e-04;
const double eta_theta_c = 1.0e-04;

const double delta_mu = 0.1;

const double epsilon = 0.1;

} /* namespace */

MinimumSolver::Lagrange::Lagrange()
{}

MinimumSolver::Lagrange::Lagrange(const VectorXd& y, const VectorXd& z)
: y(y), z(z)
{}

MinimumSolver::Options::Options()
: output(false), max_iter(100),
  residual_tol(1.0e-6), mu(0.1)
{}

MinimumSolver::Result::Result()
: converged(false), num_iter(0)
{}

MinimumSolver::Result::Result(bool converged, unsigned num_iter)
: converged(converged), num_iter(num_iter)
{}

MinimumSolver::Result::operator bool()
{
    return converged;
}

MinimumSolver::MinimumSolver()
: scaling(0, 0), dimx(0), dimy(0), dim(0),
  mu(0), delta(0), f(0), phi(0),
  alpha_max(0), alpha_z(0)
{}

void MinimumSolver::SetDimension(unsigned num_variables, unsigned num_constraints)
{
	dimx = num_variables;
	dimy = num_constraints;
	dim  = dimx + dimy;
}

void MinimumSolver::SetObjective(const Objective& objective)
{
    this->objective = objective;
}

void MinimumSolver::SetConstraint(const Constraint& constraint)
{
    this->constraint = constraint;
}

void MinimumSolver::SetScaling(const Scaling& scaling)
{
	this->scaling = scaling;
}

void MinimumSolver::SetOptions(const Options& options)
{
    this->options = options;
}

MinimumSolver::Result MinimumSolver::Solve(VectorXd& xguess)
{
    Lagrange lagrange;

    Result result = Solve(xguess, lagrange);

    return result;
}

MinimumSolver::Result MinimumSolver::Solve(VectorXd& xguess, Lagrange& lagrange)
{
    Initialise(xguess, lagrange);

    Result result;

    OutputHeader();
    while(result.num_iter <= options.max_iter and not result.converged)
    {
        OutputState(result.num_iter);
        Iterate();

        result.converged = CheckConvergence();
        result.num_iter++;

//        if(CheckSubproblemOptimality())
//            UpdatePenaltyBarrierParameters();
    }
    OutputHeader();

    xguess.noalias() = x;
    lagrange.y.noalias() = y;
    lagrange.z.noalias() = z;

    return result;
}

void MinimumSolver::Initialise(const VectorXd& xguess, const Lagrange& lagrange)
{
	// Initialise the iterate `x`
	x = xguess;

    // Initialise the Lagrange multipliers `y` and `z`
    y = lagrange.y;
    z = lagrange.z;

    // Initialise the barrier parameter `mu`
    mu = options.mu;

    // Initialise the objective function and its partial derivatives
    UpdateObjectiveFunction();

    // Initialise the constraint function and its partial derivatives
    UpdateConstraintFunction();

    // Check if a initial guess for x has been provided
    if(x.rows() != int(dimx))
        x = VectorXd::Constant(dimx, 1.0e-8);

    // Check if a initial guess for y has been provided
    if(y.rows() != int(dimy))
        y = VectorXd::Constant(dimy, 1.0e-8);

    // Check if a initial guess for z has been provided
    if(z.rows() != int(dimx))
        z = VectorXd::Constant(dimx, 1.0e-8);
}

void MinimumSolver::UpdateObjectiveFunction()
{
	std::tie(f, grad_f, hessian_f) = objective(x);
}

void MinimumSolver::UpdateConstraintFunction()
{
	std::tie(h, grad_h, hessian_h) = constraint(x);
}

void MinimumSolver::UpdateBarrierFunction()
{
    phi = f - mu * std::log(x.prod());

    grad_phi.noalias()  = grad_f;
    grad_phi.noalias() -= mu * x.cwiseInverse();
}

void MinimumSolver::UpdateExtendedHessian()
{
    H = hessian_f;

    for(unsigned j = 0; j < hessian_h.size(); ++j)
        H.noalias() += y[j] * hessian_h[j];

    H.diagonal() += z.cwiseQuotient(x);
}

void MinimumSolver::UpdateLinearSystem()
{
    rhs.resize(dim);
    lhs.resize(dim, dim);

    lhs.block(0000, 0000, dimx, dimx) = H;
    lhs.block(0000, dimx, dimx, dimy) = grad_h.transpose();
    lhs.block(dimx, 0000, dimy, dimx) = grad_h;
    lhs.block(dimx, dimx, dimy, dimy) = MatrixXd::Zero(dimy, dimy);

    rhs.segment(0000, dimx) = -grad_f.array() + mu/x.array();
    rhs.segment(dimx, dimy) = -h;

//    rhs.segment(0000, dimx) = grad_phi + grad_h.transpose() * y;
//    rhs.segment(dimx, dimy) = h;
}

void MinimumSolver::UpdateSearchDirections()
{
    // Solve the reduced KKT linear system
    solution = lhs.lu().solve(rhs);

    // Extract the Newton search directions dx and dy from solution
    dx = solution.segment(0000, dimx);
    dy = solution.segment(dimx, dimy);

    // Calculate the Newton search direction dz
    dz = (mu - x.array()*z.array() - z.array()*dx.array())/x.array();
}

void MinimumSolver::ApplyInertiaCorrection()
{
//    // If the Newton direction is a descent direction and the linear system is not ill-conditioned, skip this step
//    const bool descent = IsDescentDirection();
//    const bool ill_conditioned = IsIllConditioned();
//
////    if(IsDescentDirection() and not IsIllConditioned())
////        return;
//
//    if(descent and not ill_conditioned)
//        return;
//
//    // Update the inertia correction parameter  before the loop
//    delta = (delta == 0) ? delta_start : std::max(delta_min, kappa_minus * delta);
//
//    while(true)
//    {
//        // Correct the inertia of the coefficient matrix
//        lhs.block(0, 0, dimx, dimx).noalias() = H + delta * MatrixXd::Identity(dimx, dimx);
//
//        // Solve the linear system and compute the modified Newton directions
//        CalculateNewtonDirections();
//
//        // Check if the correction eliminated the ill-condition and non-descent state of the direction
//        if(IsDescentDirection() and not IsIllConditioned())
//            break;
//
//        // Increase the inertia correction parameter
//        delta *= (delta == delta_start) ? kappa_plus_plus : kappa_plus;
//    }
}

void MinimumSolver::ApplyAdequacyCorrection()
{
//    const double norm_grad_phi = grad_phi.norm();
//    const double norm_dx = dx.norm();
//
//    if(grad_phi.dot(dx) >= -theta * norm_grad_phi * norm_dx)
//    {
//        dx = -grad_phi;
//    }
//    else
//    {
//        if(norm_dx > beta_max * norm_grad_phi)
//            dx *= beta_max * norm_grad_phi/norm_dx;
//
//        if(norm_dx < beta_min * norm_grad_phi)
//            dx *= beta_min * norm_grad_phi/norm_dx;
//    }
}

void MinimumSolver::CalculateIterates()
{
    alpha_max = CalculateBacktrackingStepLength();
    alpha_z   = CalculateBoundaryFraction(z, dz);

    x += alpha_max * dx;
    y += alpha_max * dy;
    z += alpha_z   * dz;
}

void MinimumSolver::Iterate()
{
    UpdateObjectiveFunction();
    UpdateConstraintFunction();
    UpdateBarrierFunction();
    UpdateExtendedHessian();
    UpdateLinearSystem();
    UpdateSearchDirections();
    ApplyInertiaCorrection();
    ApplyAdequacyCorrection();
    CalculateIterates();
}

bool MinimumSolver::CheckConvergence()
{
    const double error = UpdateOptimalityError();

    return error <= options.residual_tol;
}

//bool MinimumSolver::CheckSubproblemOptimality()
//{
//    if(grad_phi.lpNorm<Infinity>() <= kappa_eps * mu))
//    {
//        y = rho * h;
//        z = mu * x.cwiseInverse();
//
//        return true;
//    }
//
//    return false;
//}

void MinimumSolver::UpdatePenaltyBarrierParameters()
{
    // Decrease mu and increase rho
    mu  = std::min(kappa_mu*mu, std::pow(mu, theta_mu));
}

bool MinimumSolver::IsIllConditioned() const
{
    return (lhs*solution - rhs).norm()/rhs.norm() > singular_eps;
}

bool MinimumSolver::IsDescentDirection() const
{
    return grad_phi.dot(dx) < 0;
}

double MinimumSolver::CalculateBoundaryFraction(const VectorXd& p, const VectorXd& dp) const
{
    const double tau = std::max(tau_min, 1 - mu);

    double alpha = 1.0;

    for(unsigned i = 0; i < dimx; ++i)
    {
        const double alphai = -tau*p[i]/dp[i];

        if(0 <= alphai and alphai <= 1.0)
            alpha = std::min(alpha, 0.98*alphai);
    }

    return alpha;
}

double MinimumSolver::CalculateBarrierFunction(const VectorXd& x) const
{
    const double f        = std::get<0>(objective(x));
    const double sum_logx = std::log(x.prod());

    return f - mu * sum_logx;
}

double MinimumSolver::CalculateBacktrackingStepLength() const
{
    const double aux = eta_phi * grad_phi.dot(dx);

    VectorXd x_new(x.rows());

    double alpha = CalculateBoundaryFraction(x, dx);

//    return alpha;

    while(alpha > alpha_eps)
    {
        x_new.noalias() = x + alpha*dx;

        const double phi_new = CalculateBarrierFunction(x_new);

        if(phi_new <= phi + alpha*aux)
            return alpha;

        alpha *= 0.90;
    }

    return 0.0;
}

void MinimumSolver::OutputHeader() const
{
    if(options.output)
    {
    	const unsigned nfill = 10 + dimx*15 + 7*15;
    	const std::string bar(nfill, '=');

    	std::cout << bar << std::endl;
        std::cout << std::setw(10) << std::left << "Iter";
        for(unsigned i = 0; i < dimx; ++i)
        {
            std::stringstream xi; xi << "x[" << i << "]";
            std::cout << std::setw(15) << std::left << xi.str();
        }
        std::cout << std::setw(15) << std::left << "f(x)";
        std::cout << std::setw(15) << std::left << "phi(x)";
        std::cout << std::setw(15) << std::left << "|h(x)|";
        std::cout << std::setw(15) << std::left << "residual";
        std::cout << std::setw(15) << std::left << "|Xz|";
        std::cout << std::setw(15) << std::left << "alpha_max";
        std::cout << std::setw(15) << std::left << "alpha_z";
        std::cout << std::setw(15) << std::left << "mu";
        std::cout << std::endl;
        std::cout << bar << std::endl;
    }
}

void MinimumSolver::OutputState(unsigned iter) const
{
    if(options.output)
    {
        auto to_string = [=](double val)
		{
    		std::stringstream ss;
    		ss.precision(std::cout.precision()); // sets `ss` with the current precision of std::cout
    		ss.setf(std::cout.flags()); // sets `ss` with the current format of std::cout
    		ss << val;
    		return (iter) ? ss.str() : "---";
		};

        const double res = iter ? residual.lpNorm<Infinity>() : 0.0;
        const double Xz  = iter ? x.cwiseProduct(z).lpNorm<Infinity>() : 0.0;

        std::cout << std::setw(10) << std::left << iter;
        for(unsigned j = 0; j < dimx; ++j)
            std::cout << std::setw(15) << std::left << x[j];
        std::cout << std::setw(15) << std::left << to_string(f);
        std::cout << std::setw(15) << std::left << to_string(phi);
        std::cout << std::setw(15) << std::left << to_string(h.norm());
        std::cout << std::setw(15) << std::left << to_string(res);
        std::cout << std::setw(15) << std::left << to_string(Xz);
        std::cout << std::setw(15) << std::left << to_string(alpha_max);
        std::cout << std::setw(15) << std::left << to_string(alpha_z);
        std::cout << std::setw(15) << std::left << mu;
        std::cout << std::endl;
    }
}

double MinimumSolver::UpdateOptimalityError()
{
    // Calculate the residual vector
    residual = grad_f + grad_h.transpose() * y - z;

    // Calculate the norms of the Lagrange multipliers y and z
    const double ynorm1 = y.lpNorm<1>();
    const double znorm1 = z.lpNorm<1>();

    // Calculate the scaling parameters
    const double sd = std::max(s_max, (ynorm1 + znorm1)/dim)/s_max;
    const double sc = std::max(s_max, znorm1/dim)/s_max;

    // Calculate the feasibility, centrality and optimality errors
    const double error_f = h.lpNorm<Infinity>();
    const double error_c = (x.array() * z.array() - mu).matrix().lpNorm<Infinity>();
    const double error_o = residual.lpNorm<Infinity>();

    return std::max(error_f, std::max(error_c/sc, error_o/sd));
}

void MinimumSolver::UpdateBarrierParameter()
{
    mu = delta_mu * x.dot(z)/(2*dim);
}

} /* namespace Optima */
