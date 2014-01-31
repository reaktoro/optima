/*
 * IPFilterErrors.hpp
 *
 *  Created on: 5 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <stdexcept>

namespace Optima {

struct IPFilterErrorInitialGuess : public std::exception {};
struct IPFilterErrorSearchDelta  : public std::exception {};
struct IPFilterErrorIteration    : public std::exception {};

struct IPFilterErrorInitialGuessFloatingPoint : public IPFilterErrorInitialGuess
{
    virtual const char* what() const throw()
    {
        return "Unable to proceed with the calculation using the provided initial "
            "guess. This initial guess results in a IEEE floating-point exception "
            "when either the objective or the constraint function is evaluated. "
            "Try another initial guess.";
    }
};

struct IPFilterErrorInitialGuessActivePartition : public IPFilterErrorInitialGuess
{
    virtual const char* what() const throw()
    {
        return "Unable to proceed with given initial guess. The monitoring of the "
           "iterations so far indicates that there are active partitions at the "
           "provided initial guess that should in fact be inactive. Try another "
           "initial guess with such partitions not active.";
    }
};

struct IPFilterErrorSearchDeltaNeighborhood : public IPFilterErrorSearchDelta
{
    virtual const char* what() const throw()
    {
        return "Unable to find a trust-region radius that satisfies the centrality "
            "neighborhood condition in the interior-point algorithm. Try another "
            "initial guess or decrease the minimum allowed delta.";
    }
};

struct IPFilterErrorSearchDeltaTrustRegion : public IPFilterErrorSearchDelta
{
    virtual const char* what() const throw()
    {
        return "Unable to find a trust-region radius that satisfies the Cauchy "
           "condition of sufficient decrease in the interior-point algorithm. "
           "Try another initial guess or decrease the minimum allowed delta.";
    }
};

struct IPFilterErrorSearchDeltaTrustRegionRestoration : public IPFilterErrorSearchDelta
{
    virtual const char* what() const throw()
    {
        return "Unable to find a trust-region radius that satisfies the Cauchy "
           "condition of sufficient decrease in the restoration algorithm. "
           "Try another initial guess or decrease the minimum allowed delta.";
    }
};

struct IPFilterErrorIterationAttemptLimit : public IPFilterErrorIteration
{
    virtual const char* what() const throw()
    {
        return "Unable to converge to an optimum point within the specified maximum "
            "number of iterations. Try another initial guess or increase the allowed "
            "maximum number of iterations.";
    }
};

} /* namespace Optima */
