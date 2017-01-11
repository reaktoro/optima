// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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
#include <Optima/Common/Index.hpp>
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// The parameters of an optimization problem that change with more frequency.
struct OptimumParams
{
    /// The right-hand side vector of the linear equality constraint \eq{Ax = a}.
    VectorXd a;

    /// The right-hand side vector of the linear equality constraint \eq{Bx \geq b}.
    VectorXd b;

    /// The lower bounds of the variables \eq{x}.
    VectorXd xlower;

    /// The upper bounds of the variables \eq{x}.
    VectorXd xupper;

    /// The values of the variables in \eq{x} that are fixed.
    VectorXd xfixed;

    /// The indices of the variables in \eq{x} that are fixed.
    Indices ifixed;
};

} // namespace Optima
