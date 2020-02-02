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

namespace Optima {

/// The state of the optimization variables.
struct State
{
    /// The variables \eq{x} of the optimization problem.
    Vector x;

    /// The Lagrange multipliers \eq{y} of the optimization problem.
    Vector y;

    /// The instability measures of variables \eq{x} defined as \eq{z = g + W^{T}y}.
    Vector z;

    /// The sensitivity derivatives \eq{\partial x/\partial p} with respect to parameters \eq{p}.
    Matrix dxdp;

    /// The sensitivity derivatives \eq{\partial y/\partial p} with respect to parameters \eq{p}.
    Matrix dydp;

    /// The sensitivity derivatives \eq{\partial z/\partial p} with respect to parameters \eq{p}.
    Matrix dzdp;
};

} // namespace Optima
