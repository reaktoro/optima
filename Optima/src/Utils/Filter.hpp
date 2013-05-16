/*
 * Filter.hpp
 *
 *  Created on: 22 Mar 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <iostream>
#include <list>
#include <vector>

namespace Optima {

class Filter
{
public:
    Filter();

    bool IsAcceptable(const std::vector<double>& point) const;

    void Add(const std::vector<double>& point);

    const std::list<std::vector<double>>& GetPoints() const;

private:
    std::list<std::vector<double>> points;
};

std::ostream& operator<<(std::ostream& out, const Filter& filter);

} /* namespace Optima */
