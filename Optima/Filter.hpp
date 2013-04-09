/*
 * Filter.hpp
 *
 *  Created on: 22 Mar 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <list>
#include <tuple>
#include <vector>

namespace Optima {

class Filter
{
public:
    Filter();

    bool IsAcceptable(const std::vector<double>& point) const;

    void Add(const std::vector<double>& point);

private:
    std::list<std::vector<double>> points;
};

} /* namespace Optima */
