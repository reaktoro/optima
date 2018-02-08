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

#pragma once

// C++ includes
#include <map>

// Optima includes
#include <Optima/Common/Index.hpp>
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// A type used to describe a system of linear constraints in an optimization problem.
struct Constraint
{
    /// The coefficient matrix of the linear equality constraint \eq{Ax = a}.
    MatrixXd A;

    /// The right-hand side vector of the linear equality constraint \eq{Ax = a}.
    VectorXd a;

    /// The coefficient matrix of the linear inequality constraint \eq{Bx \geq b}.
    MatrixXd B;

    /// The right-hand side vector of the linear equality constraint \eq{Bx \geq b}.
    VectorXd b;

    /// The lower bound of the primal variables \eq{x}.
    VectorXd xlower;

    /// The upper bound of the primal variables \eq{x}.
    VectorXd xupper;

    /// The indices of the variables in \eq{x} that are fixed at given values.
    Indices ifixed;

    /// The values of the fixed variables in \eq{x}.
    VectorXd xfixed;
};

} // namespace Optima
