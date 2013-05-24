/*
 * QualitySolver.cpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#include "QualitySolver.hpp"

// C++ includes
#include <cmath>
#include <iostream>

// Optima includes
#include <Optima/Misc/BrentSolver.hpp>
#include <Optima/Misc/GoldenSolver.hpp>
#include <Optima/Misc/Utils.hpp>
#include <Optima/Utils/Functions.hpp>

namespace Optima {

QualitySolver::QualitySolver()
{}

void QualitySolver::SetData(const Data& data)
{
    this->data = data;
}

void QualitySolver::SetParams(const Params& params)
{
    this->params = params;
}

const QualitySolver::Data& QualitySolver::GetData() const
{
    return data;
}

QualitySolver::Data& QualitySolver::GetData()
{
    return data;
}

double QualitySolver::CalculateQuality(double sigma)
{
    const auto& x   = data.x;
    const auto& y   = data.y;
    const auto& z   = data.z;
    const auto& dx0 = data.dx0;
    const auto& dy0 = data.dy0;
    const auto& dz0 = data.dz0;
    const auto& dx1 = data.dx1;
    const auto& dy1 = data.dy1;
    const auto& dz1 = data.dz1;
    const auto& thh = data.thh;
    const auto& thl = data.thl;

    dxs.noalias() = dx0 + sigma*(dx1 - dx0);
    dys.noalias() = dy0 + sigma*(dy1 - dy0);
    dzs.noalias() = dz0 + sigma*(dz1 - dz0);

    const double alphax = CalculateLargestBoundaryStep(x, dxs);
    const double alphaz = CalculateLargestBoundaryStep(z, dzs);

    return std::pow((1 - alphaz)*thl, 2) + std::pow((1 - alphax)*thh, 2) +
        (x + alphax*dxs).cwiseProduct((z + alphaz*dzs)).squaredNorm();
}

double QualitySolver::CalculateSigma()
{
    auto Q = [&](double sigma)
    {
        return CalculateQuality(sigma);
    };

    const unsigned n = data.x.rows();

    const auto& x   = data.x;
    const auto& y   = data.y;
    const auto& z   = data.z;
    const auto& dx0 = data.dx0;
    const auto& dy0 = data.dy0;
    const auto& dz0 = data.dz0;

    const double alphax = CalculateLargestBoundaryStep(x, dx0);
    const double alphaz = CalculateLargestBoundaryStep(z, dz0);

    double a = params.sigma_min;
    double b = params.sigma_max;
    double c;

    Bracket(Q, a, b, c);

    BrentSolver brent;
    brent.SetFunction(Q);
    brent.Solve(a, b, c);

    return brent.GetResult().xmin;
}

} /* namespace Optima */
