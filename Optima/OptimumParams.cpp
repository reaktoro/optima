// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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

#include "OptimumParams.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/OptimumStructure.hpp>

namespace Optima {

OptimumParams::OptimumParams(const OptimumStructure& structure)
: b(structure.numEqualityConstraints()),
  xlower(structure.variablesWithLowerBounds().size()),
  xupper(structure.variablesWithUpperBounds().size()),
  xfixed(structure.variablesWithFixedValues().size())
{
    b.fill(0.0);
    xlower.fill(0.0);
    xupper.fill(0.0);
    xfixed.fill(0.0);

    objective = [](VectorConstRef)
    {
        RuntimeError("Could not evaluate the objective function.",
            "Did you forget to set the objective function in "
            "OptimumParams::objective?");

        return ObjectiveResult();
    };
}

} // namespace Optima
