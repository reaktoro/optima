/*
 * BrentSolver.cpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#include "BrentSolver.hpp"

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
 * The Brent algorithm from Numerical Recipes in C
 */
bool Brent(double ax, double bx, double cx, const Function& f,
    double tol, unsigned itmax, double& xmin, double& fmin,
    unsigned& niters, unsigned& nfunc_evals)
{
    const double CGOLD = 0.3819660;
    const double ZEPS  = 1.0e-10;

    niters = 0;
    nfunc_evals = 0;

    double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
    double e = 0.0;
    a = (ax < cx ? ax : cx);
    b = (ax > cx ? ax : cx);
    x = w = v = bx;
    fw = fv = fx = f(x);
    ++nfunc_evals;
    for(niters = 1; niters <= itmax; niters++)
    {
        xm = 0.5 * (a + b);
        tol2 = 2.0 * (tol1 = tol * std::abs(x) + ZEPS);
        if(std::abs(x - xm) <= (tol2 - 0.5 * (b - a)))
        {
            xmin = x;
            fmin = fx;
            return true;
        }
        if(std::abs(e) > tol1)
        {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if(q > 0.0) p = -p;
            q = std::abs(q);
            etemp = e;
            e = d;
            if(std::abs(p) >= std::abs(0.5 * q * etemp) or p <= q * (a - x) or p >= q * (b - x))
                d = CGOLD * (e = (x >= xm ? a - x : b - x));
            else
            {
                d = p / q;
                u = x + d;
                if(u - a < tol2 || b - u < tol2)
                    d = Sign(tol1, xm - x);
            }
        }
        else
        {
            d = CGOLD * (e = (x >= xm ? a - x : b - x));
        }
        u = std::abs(d) >= tol1 ? x + d : x + Sign(tol1, d);
        fu = f(u);
        ++nfunc_evals;
        if(fu <= fx)
        {
            if(u >= x) a = x;
            else b = x;
            Shift(v, w, x, u);
            Shift(fv, fw, fx, fu);
        }
        else
        {
            if(u < x) a = u;
            else b = u;
            if(fu <= fw || w == x)
            {
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            }
            else if(fu <= fv || v == x || v == w)
            {
                v = u;
                fv = fu;
            }
        }
    }

    return false;
}

} /* namespace Internal */

using Internal::Brent;

BrentSolver::BrentSolver()
: f([](double) {return 0.0;} )
{}

void BrentSolver::SetOptions(const Options& options)
{
    this->options = options;
}

void BrentSolver::SetFunction(const Function& f)
{
    this->f = f;
}

const BrentSolver::Options& BrentSolver::GetOptions() const
{
    return options;
}

const BrentSolver::Function& BrentSolver::GetFuction() const
{
    return f;
}

const BrentSolver::Result& BrentSolver::GetResult() const
{
    return result;
}

const BrentSolver::Statistics& BrentSolver::GetStatistics() const
{
    return statistics;
}

void BrentSolver::Solve(double a, double b, double c)
{
    bool successful = Internal::Brent(a, b, c, f, options.tolerance, options.itmax,
        result.xmin, result.fmin, statistics.niters, statistics.nfunc_evals);

    if(not successful)
        throw Error();
}

const char* BrentSolver::Error::what() const throw()
{
    return "Unable to find a minimum for the specified one-dimendional function using the Brent algorithm. "
        "Try another initial guess.";
}

} /* namespace Optima */
