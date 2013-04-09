/*
 * Scaling.cpp
 *
 *  Created on: 30 Jan 2013
 *      Author: allan
 */

#include "Scaling.hpp"

namespace Optima {

Scaling::Scaling(unsigned num_variables, unsigned num_constraints)
: num_variables(num_variables), num_constraints(num_constraints), cf(1.0),
  ch(VectorXd::Ones(num_constraints)), cx(VectorXd::Ones(num_variables))
{}

void Scaling::SetScaleObjective(double cf)
{
	this->cf = std::abs(cf);
}

void Scaling::SetScaleConstraint(const VectorXd& ch)
{
	this->ch = ch;
}

void Scaling::SetScaleVariables(const VectorXd& cx)
{
	this->cx = cx;
}

double Scaling::GetScaleObjective() const
{
	return cf;
}

const VectorXd& Scaling::GetScaleConstraint() const
{
	return ch;
}

const VectorXd& Scaling::GetScaleVariables() const
{
	return cx;
}

VectorXd Scaling::Scale(const VectorXd& x) const
{
	return cx.asDiagonal().inverse() * x;
}

VectorXd Scaling::Unscale(const VectorXd& w) const
{
	return cx.asDiagonal() * w;
}

Objective ScaledObjective(const Objective& objective, const Scaling& scaling)
{
	VectorXd x;

	double f = 0.0;

	VectorXd grad_f;

	MatrixXd hessian_f;

	auto scaled_objective = [=, &objective](const VectorXd& w) mutable
	{
		// Auxiliary references
		const auto& cf = scaling.GetScaleObjective();
		const auto& cx = scaling.GetScaleVariables();

		const auto Cx = cx.asDiagonal();

		// Calculate the iterate `x` from the scaled iterate `w`
		x.noalias() = Cx * w;

		// Evaluate the objective function
		std::tie(f, grad_f, hessian_f) = objective(x);

		// Apply the scaling transformations
		f                   = cf * f;
		grad_f   .noalias() = cf * Cx * grad_f;
		hessian_f.noalias() = cf * Cx * hessian_f * Cx;

		return std::make_tuple(f, grad_f, hessian_f);
	};

	return scaled_objective;
}

Objective ScaledObjectiveRef(const Objective& objective, const Scaling& scaling)
{
	VectorXd x;

	double f = 0.0;

	VectorXd grad_f;

	MatrixXd hessian_f;

	auto scaled_objective = [=, &objective](const VectorXd& w) mutable
	{
		// Auxiliary references
		const auto& cf = scaling.GetScaleObjective();
		const auto& cx = scaling.GetScaleVariables();

		const auto Cx = cx.asDiagonal();

		// Calculate the iterate `x` from the scaled iterate `w`
		x.noalias() = Cx * w;

		// Evaluate the objective function
		std::tie(f, grad_f, hessian_f) = objective(x);

		// Apply the scaling transformations
		f                   = cf * f;
		grad_f   .noalias() = cf * Cx * grad_f;
		hessian_f.noalias() = cf * Cx * hessian_f * Cx;

		return std::make_tuple(f, grad_f, hessian_f);
	};

	return scaled_objective;
}

Constraint ScaledConstraint(const Constraint& constraint, const Scaling& scaling)
{
	VectorXd x;

	VectorXd h;

	MatrixXd grad_h;

	std::vector<MatrixXd> hessians_h;

	auto scaled_constraint = [=](const VectorXd& w) mutable
	{
		// Auxiliary references
		const auto& ch = scaling.GetScaleConstraint();
		const auto& cx = scaling.GetScaleVariables();

		const auto Ch = ch.asDiagonal();
		const auto Cx = cx.asDiagonal();

		// Calculate the iterate `x` from the scaled iterate `w`
		x.noalias() = Cx * w;

		// Evaluate the objective function
		std::tie(h, grad_h, hessians_h) = constraint(x);

		// Apply the scaling transformations
		h = Ch * h;
		grad_h.noalias() = Ch * grad_h * Cx;
		for(unsigned i = 0; i < hessians_h.size(); ++i)
			hessians_h[i].noalias() = ch[i] * Cx * hessians_h[i] * Cx;

		return std::make_tuple(h, grad_h, hessians_h);
	};

	return scaled_constraint;
}

Constraint ScaledConstraintRef(const Constraint& constraint, const Scaling& scaling)
{
	VectorXd x;

	VectorXd h;

	MatrixXd grad_h;

	std::vector<MatrixXd> hessians_h;

	auto scaled_constraint = [=, &constraint](const VectorXd& w) mutable
	{
		// Auxiliary references
		const auto& ch = scaling.GetScaleConstraint();
		const auto& cx = scaling.GetScaleVariables();

		const auto Ch = ch.asDiagonal();
		const auto Cx = cx.asDiagonal();

		// Calculate the iterate `x` from the scaled iterate `w`
		x.noalias() = Cx * w;

		// Evaluate the objective function
		std::tie(h, grad_h, hessians_h) = constraint(x);

		// Apply the scaling transformations
		h.noalias() = Ch * h;
		grad_h.noalias() = Ch * grad_h * Cx;
		for(unsigned i = 0; i < hessians_h.size(); ++i)
			hessians_h[i].noalias() = ch[i] * Cx * hessians_h[i] * Cx;

		return std::make_tuple(h, grad_h, hessians_h);
	};

	return scaled_constraint;
}

} /* namespace Optima */

