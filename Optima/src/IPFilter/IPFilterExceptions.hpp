/*
 * IPFilterExceptions.hpp
 *
 *  Created on: 5 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <stdexcept>
#include <sstream>

namespace Optima {

struct MaxIterationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Unable to converge to an optimum point within the specified maximum "
               "number of iterations. Try another initial guess or increase "
               "the allowed maximum number of iterations.";
    }
};

struct InitialGuessError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Unable to proceed with the calculation using the provided initial "
               "guess. This initial guess results in a IEEE floating-point exception "
               "when either the objective or the constraint function is evaluated. Try "
               "another initial guess.";
    }
};

struct SearchDeltaNeighborhoodError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Could not find a trust-region radius that satisfies the centrality "
               "neighborhood condition in the interior-point algorithm. Try another "
               "initial guess or decrease the minimum allowed delta.";
    }
};

struct SearchDeltaTrustRegionError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Could not find a trust-region radius that satisfies the Cauchy "
               "condition of sufficient decrease in the interior-point algorithm. "
               "Try another initial guess or decrease the minimum allowed delta.";
    }
};

struct SearchDeltaTrustRegionRestorationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Could not find a trust-region radius that satisfies the Cauchy "
               "condition of sufficient decrease in the restoration algorithm. "
               "Try another initial guess or decrease the minimum allowed delta.";
    }
};

} /* namespace Optima */


