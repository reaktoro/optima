// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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
#include <Optima/ResourcesFunction.hpp>
#include <Optima/TransformFunction.hpp>

namespace Optima {

/// Used to represent a master optimization problem.
struct MasterProblem
{
    MasterDims dims;       ///< The dimensions of the master variables.
    ResourcesFunction r;   ///< The optional function that precomputes shared resources for objective and constraint functions.
    ObjectiveFunction f;   ///< The objective function *f(x, p)*.
    ConstraintFunction h;  ///< The nonlinear equality constraint function *h(x, p)*.
    ConstraintFunction v;  ///< The external nonlinear constraint function *v(x, p)*.
    Matrix Ax;             ///< The matrix *Ax* in *W = [Ax Ap; Jx Jp]*.
    Matrix Ap;             ///< The matrix *Ap* in *W = [Ax Ap; Jx Jp]*.
    Vector b;              ///< The right-hand side vector *b* in the linear equality constraints.
    Vector xlower;         ///< The lower bounds for variables *x*.
    Vector xupper;         ///< The upper bounds for variables *x*.
    Vector plower;         ///< The lower bounds for variables *p*.
    Vector pupper;         ///< The upper bounds for variables *p*.
    TransformFunction phi; ///< The custom variable transformation function.
    Vector c;              ///< The sensitivity parameters *c*.
    Matrix bc;             ///< The Jacobian matrix of *b* with respect to the sensitivity parameters *c*.
};

} // namespace Optima
