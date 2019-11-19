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

// Optima includes
#include <Optima/Matrix.hpp>
#include <Optima/Constraints.hpp>
#include <Optima/PrimalVariables.hpp>
#include <Optima/LagrangeMultipliers.hpp>
#include <Optima/ComplementarityVariables.hpp>

namespace Optima {

/// Used to describe the state of an optimization calculation.
class State
{
public:
    explicit State(const Constraints& constraints)
    : x(constraints), y(constraints), z(constraints)
    {}

// private:
    /// The primal variables of the optimization problem.
    PrimalVariables x;

    /// The Lagrange multipliers of the optimization problem.
    LagrangeMultipliers y;

    /// The complementarity variables of the optimization problem.
    ComplementarityVariables z;
    // /// The primal solution of the optimization problem.
    // Vector x;

    // /// The dual solution of the optimization problem with respect to the equality constraints.
    // Vector y;

    // /// The dual solution of the optimization problem with respect to the lower bound constraints.
    // Vector z;

    // /// The dual solution of the optimization problem with respect to the upper bound constraints.
    // Vector w;
};

} // namespace Optima
