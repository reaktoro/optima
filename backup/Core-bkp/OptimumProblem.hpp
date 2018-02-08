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
#include <Optima/Math/Matrix.hpp>
#include <Optima/Core/Hessian.hpp>
#include <Optima/Core/Objective.hpp>

namespace Optima {

/// A type that describes the non-linear constrained optimization problem
struct OptimumProblem
{
    /// Construct a default OptimumProblem instance.
    OptimumProblem();

    /// Construct an OptimumProblem instance with given number of variables.
    /// @param n The number of variables in the optimization problem.
    OptimumProblem(Index n);

    /// The number of primal variables `x`
    Index n;

    /// The objective function.
    ObjectiveFunction objective;

    /// The coefficient vector of a linear programming problem.
    Vector c;

    /// The coefficient matrix of the linear equality constraint @f$Ax = a@f$.
    Matrix A;

    /// The right-hand side vector of the linear equality constraint @f$Ax = a@f$.
    Vector a;

    /// The coefficient matrix of the linear inequality constraint @f$Bx \geq b@f$.
    Matrix B;

    /// The right-hand side vector of the linear equality constraint @f$Bx \geq b@f$.
    Vector b;

    /// The lower bound of the primal variables `x`.
    Vector xlower;

    /// The upper bound of the primal variables `x`.
    Vector xupper;

    /// The values of the fixed variables.
    std::map<Index, double> xfixed;
};

} // namespace Optima
