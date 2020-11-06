// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#pragma once

// C++ includes
#include <vector>

// Optima includes
#include <Optima/Index.hpp>

namespace Optima {

/// A type that contains data for convergence analysis of an optimization calculation.
class Analysis
{
public:
    /// Construct a default Analysis object.
    Analysis() = default;

    /// Initialize this Analysis object with given maximum number of iterations.
    auto initialize(Index maxiterations) -> void
    {
        L.clear(); L.reserve(maxiterations);
        E.clear(); E.reserve(maxiterations);
    }

    /// Compute the convergence rate of the optimization calculation.
    auto convergenceRate() const -> double
    {
        using std::log;
        const auto i = E.size() - 1;
        if(i >= 2)
            return log(E[i] / E[i-1]) / log(E[i-1] / E[i-2]);
        else return 1.0;
    }

    /// The evaluated Lagrange function *L* at each iteration.
    std::vector<double> L;

    /// The computed error *E = ||grad(L)||^2* at each iteration.
    std::vector<double> E;
};

} // namespace Optima
