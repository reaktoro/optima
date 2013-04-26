/*
 * Scaling.cpp
 *
 *  Created on: 23 Apr 2013
 *      Author: allan
 */

#include "Scaling.hpp"

namespace Optima {

Scaling::Scaling()
: Df(1.0)
{}

void Scaling::SetScalingVariables(const VectorXd& Dx)
{
    this->Dx = Dx;
}

void Scaling::SetScalingConstraint(const VectorXd& Dh)
{
    this->Dh = Dh;
}

void Scaling::SetScalingObjective(double Df)
{
    this->Df = Df;
}

const VectorXd& Scaling::GetScalingVariables() const
{
    return Dx;
}

const VectorXd& Scaling::GetScalingConstraint() const
{
    return Dh;
}

double Scaling::GetScalingObjective() const
{
    return Df;
}

bool Scaling::HasScalingVariables() const
{
    return Dx.rows();
}

bool Scaling::HasScalingConstraint() const
{
    return Dh.rows();
}

bool Scaling::HasScalingObjective() const
{
    return Df != 1.0;
}

void Scaling::ScaleX(VectorXd& x) const
{
    if(HasScalingVariables())
        x.array() /= Dx.array();
}

void Scaling::ScaleY(VectorXd& y) const
{
}

void Scaling::ScaleZ(VectorXd& z) const
{
    if(HasScalingVariables())
        z.array() *= Dx.array();
}

void Scaling::ScaleConstraint(ConstraintResult& h) const
{
    if(HasScalingVariables())
    {
        h.grad = h.grad * Dx.asDiagonal();

        for(unsigned i = 0; i < h.hessian.size(); ++i)
            h.hessian[i] = Dx.asDiagonal() * h.hessian[i] * Dx.asDiagonal();
    }

    if(HasScalingConstraint())
    {
        h.func = Dh.asDiagonal().inverse() * h.func;
        h.grad = Dh.asDiagonal().inverse() * h.grad;

        for(unsigned i = 0; i < h.hessian.size(); ++i)
            h.hessian[i] *= 1.0/Dh[i];
    }
}

void Scaling::ScaleObjective(ObjectiveResult& f) const
{
    if(HasScalingVariables())
    {
        f.grad    = Dx.asDiagonal() * f.grad;
        f.hessian = Dx.asDiagonal() * f.hessian * Dx.asDiagonal();
    }

    if(HasScalingObjective())
    {
        f.func    /= Df;
        f.grad    /= Df;
        f.hessian /= Df;
    }
}

void Scaling::UnscaleX(VectorXd& x) const
{
    if(HasScalingVariables())
        x.array() *= Dx.array();
}

void Scaling::UnscaleY(VectorXd& y) const
{
}

void Scaling::UnscaleZ(VectorXd& z) const
{
    if(HasScalingVariables())
        z.array() /= Dx.array();
}

void Scaling::UnscaleConstraint(ConstraintResult& h) const
{
    if(HasScalingVariables())
    {
        h.grad = h.grad * Dx.asDiagonal().inverse();

        for(unsigned i = 0; i < h.hessian.size(); ++i)
            h.hessian[i] = Dx.asDiagonal().inverse() * h.hessian[i] * Dx.asDiagonal().inverse();
    }

    if(HasScalingConstraint())
    {
        h.func = Dh.asDiagonal() * h.func;
        h.grad = Dh.asDiagonal() * h.grad;

        for(unsigned i = 0; i < h.hessian.size(); ++i)
            h.hessian[i] *= Dh[i];
    }
}

void Scaling::UnscaleObjective(ObjectiveResult& f) const
{
    if(HasScalingVariables())
    {
        f.grad.array() /= Dx.array();
        f.hessian = Dx.asDiagonal().inverse() * f.hessian * Dx.asDiagonal().inverse();
    }

    if(HasScalingObjective())
    {
        f.func    *= Df;
        f.grad    *= Df;
        f.hessian *= Df;
    }
}

} /* namespace Optima */

