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
        return "The algorithm was unable to converge to an optimum point "
               "within the maximum specified maximum number of iterations. "
               "Try to use a another initial guess or increase the maximum "
               "allowed number of iterations.";
    }
};

struct MaxIterationRestorationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "The restoration algorithm was unable to converge to a suitable "
               "point within the maximum specified maximum number of iterations. "
               "Try to use a another initial guess or increase the maximum "
               "allowed number of iterations.";
    }
};

struct NeighborhoodError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "The algorithm could not find a trust-region radius that "
               "satisfies the centrality neighborhood condition. Try to use a "
               "another initial guess or set the minimum allowed delta to a "
               "smaller value.";
    }
};

struct NeighborhoodRestorationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "The restoration algorithm could not find a trust-region radius that "
               "satisfies the centrality neighborhood condition. Try to use a "
               "another initial guess or set the minimum allowed delta to a "
               "smaller value.";
    }
};

struct TrialTestError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "The main algorithm could not find a trust-region radius that "
               "satisfies the trial tests. Try to use a another initial guess "
               "or set the minimum allowed delta to a smaller value.";
    }
};

struct TrialTestRestorationError : public std::exception
{
    virtual const char* what() const throw()
    {
        return "The restoration algorithm could not find a trust-region radius that "
               "satisfies the trial tests. Try to use a another initial guess "
               "or set the minimum allowed delta to a smaller value.";
    }
};

} /* namespace IPFilter */
} /* namespace Optima */


