/*
 * Filter.cpp
 *
 *  Created on: 22 Mar 2013
 *      Author: allan
 */

#include "Filter.hpp"

// C++ includes
#include <algorithm>
#include <cmath>

namespace Optima {

bool operator<(const std::vector<double>& a, const std::vector<double>& b)
{
    for(unsigned i = 0; i < a.size(); ++i)
        if(b[i] < a[i]) return false;
    return true;
}

bool operator>(const std::vector<double>& a, const std::vector<double>& b)
{
    return b < a;
}

Filter::Filter()
{}

bool Filter::IsAcceptable(const std::vector<double>& point) const
{
    for(const std::vector<double>& entry : points)
        if(point > entry)
            return false;
    return true;
}

void Filter::Add(const std::vector<double>& point)
{
    // Define the domination function to remove dominated points from the filter
    auto dominated = [=](const std::vector<double>& entry)
    {
        return point < entry;
    };

    // Remove all dominated points in the filter
    std::remove_if(points.begin(), points.end(), dominated);

    // Add the new point to the filter
    points.push_back(point);
}

} /* namespace Optima */
