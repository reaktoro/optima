/*
 * Utils.hpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <cmath>
#include <functional>

namespace Optima {

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
 * Computes the magniture of @c a times sign of @c b
 */
inline double Sign(double a, double b)
{
    return (b >= 0) ? std::abs(a) : -std::abs(a);
}

/**
 * Finds a third point @e c in order to bracket the minimum of the function @e f
 */
void Bracket(const std::function<double(double)>& f, double& a, double& b, double& c, unsigned* niters=0);

} /* namespace Optima */
