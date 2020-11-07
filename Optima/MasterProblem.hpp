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
#include <Optima/ConstraintFunction.hpp>
#include <Optima/MasterDims.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/ObjectiveFunction.hpp>
#include <Optima/TransformFunction.hpp>

namespace Optima {

/// Used to represent a master optimization problem.
struct MasterProblem
{
    ObjectiveFunction const& f;   ///< The objective function *f(x, p)*.
    ConstraintFunction const& h;  ///< The nonlinear equality constraint function *h(x, p)*.
    ConstraintFunction const& v;  ///< The external nonlinear constraint function *v(x, p)*.
    VectorConstRef const& b;      ///< The right-hand side vector b in the linear equality constraints.
    VectorConstRef const& xlower; ///< The lower bounds for variables *x*.
    VectorConstRef const& xupper; ///< The upper bounds for variables *x*.
    TransformFunction const& phi; ///< The custom variable transformation function.
};

} // namespace Optima
