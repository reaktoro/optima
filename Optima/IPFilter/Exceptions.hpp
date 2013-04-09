/*
 * Exceptions.hpp
 *
 *  Created on: 5 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <stdexcept>
#include <sstream>

namespace Optima {
namespace IPFilter {

struct MaxIterationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Unable to converge to an optimum point within the specified maximum "
               "number of iterations. Try to use a another initial guess or increase "
               "the allowed maximum number of iterations.";
    }
};

struct SearchDeltaNeighborhoodError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Could not find a trust-region radius that satisfies the centrality "
               "neighborhood condition in the interior-point algorithm. Try to use "
               "a another initial guess or set the minimum allowed delta to a "
               "smaller value.";
    }
};

struct SearchDeltaError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Could not find a trust-region radius that satisfies the Cauchy "
               "condition of sufficient decrease in the interior-point algorithm. "
               "Try to use a another initial guess or set the minimum allowed delta "
               "to a smaller value.";
    }
};

struct SearchDeltaRestorationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "Could not find a trust-region radius that satisfies the Cauchy "
               "condition of sufficient decrease in the restoration algorithm. "
               "Try to use a another initial guess or set the minimum allowed "
               "delta to a smaller value.";
    }
};

} /* namespace IPFilter */
} /* namespace Optima */


