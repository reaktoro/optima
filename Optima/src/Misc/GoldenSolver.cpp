/*
 * GoldenSolver.cpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#include "GoldenSolver.hpp"

// C++ includes
#include <algorithm>
#include <cmath>
#include <iostream>

namespace Optima {
namespace Internal {

/**
 * Shifts @c a to @c b, and @c b to @c c
 */
inline void Shift(double& a, double& b, double c)
{
    a = b;
    b = c;
}

/**
 * Shifts @c a to @c b, @c b to @c c, and @c c to @c d
 */
inline void Shift(double& a, double& b, double& c, double d)
{
    a = b;
    b = c;
    c = d;
}

/**
 * Computes the magniture of @c a times Sign of @c b
 */
inline double Sign(double a, double b)
{
    return (b >= 0) ? std::abs(a) : -std::abs(a);
}

/**
 * The one-dimensional function signature
 */
typedef std::function<double(double)> Function;

/**
 * The Golden algorithm from Numerical Recipes in C
 */
bool Golden(double a, double b, double c, const Function& f, double tol,
    unsigned itmax, double& xmin, double& fmin, unsigned& niters, unsigned& nfunc_evals)
{
    const double RGOLD = 0.61803399;
    const double CGOLD = 1.0 - RGOLD;

    double f1, f2, x0, x1, x2, x3;

    x0 = a;
    x3 = c;
    if(std::abs(c - b) > std::abs(b - a))
    {
        x1 = b;
        x2 = b + CGOLD * (c - b);
    }
    else
    {
        x2 = b;
        x1 = b - CGOLD * (b - a);
    }
    f1 = f(x1);
    f2 = f(x2);

    niters = 0;
    nfunc_evals = 0;

    while(std::abs(x3 - x0) > tol * (std::abs(x1) + std::abs(x2)))
    {
        if(f2 < f1)
        {
            Shift(x0, x1, x2, RGOLD*x1 + CGOLD*x3);
            Shift(f1, f2, f(x2));
        }
        else
        {
            Shift(x3, x2, x1, RGOLD*x2 + CGOLD*x0);
            Shift(f2, f1, f(x1));
        }
        ++nfunc_evals;
        ++niters;

        if(niters > itmax)
            return false;
    }

    if(f1 < f2)
    {
        xmin = x1;
        fmin = f1;
    }
    else
    {
        xmin = x2;
        fmin = f2;
    }

    return true;
}

} /* namespace Internal */

using Internal::Golden;

GoldenSolver::GoldenSolver()
: f([](double) {return 0.0;} )
{}

void GoldenSolver::SetOptions(const Options& options)
{
    this->options = options;
}

void GoldenSolver::SetFunction(const Function& f)
{
    this->f = f;
}

const GoldenSolver::Options& GoldenSolver::GetOptions() const
{
    return options;
}

const GoldenSolver::Function& GoldenSolver::GetFuction() const
{
    return f;
}

const GoldenSolver::Result& GoldenSolver::GetResult() const
{
    return result;
}

const GoldenSolver::Statistics& GoldenSolver::GetStatistics() const
{
    return statistics;
}

void GoldenSolver::Solve(double a, double b, double c)
{
    bool successful = Internal::Golden(a, b, c, f, options.tolerance, options.itmax,
        result.xmin, result.fmin, statistics.niters, statistics.nfunc_evals);

    if(not successful)
        throw Error();
}

const char* GoldenSolver::Error::what() const throw()
{
    return "Unable to find a minimum for the specified one-dimendional function using the Golden algorithm. "
        "Try another initial guess.";
}

} /* namespace Optima */
