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
#include <functional>

// Optima includes
#include <Optima/Matrix.hpp>

namespace Optima {

/// The requirements in the evaluation of the objective function.
struct ObjectiveRequirement
{
    /// The boolean flag indicating the objective value is needed.
    bool f = true;

    /// The boolean flag indicating the objective gradient is needed.
    bool g = true;

    /// The boolean flag indicating the objective Hessian is needed.
    bool H = true;
};

/// The result of the evaluation of an objective function.
/// @see ObjectiveFunction
struct ObjectiveResult
{
    /// The evaluated value of the objective function.
    double& f;

    /// The evaluated gradient of the objective function.
    VectorRef g;

    /// The evaluated Hessian of the objective function.
    MatrixRef H;

    /// The requirements in the evaluation of the objective function.
    ObjectiveRequirement requires;

    /// The boolean flag that indicates if the objective function evaluation failed.
    bool failed = false;

    /// Construct an ObjectiveResult object with given objective value, gradient and Hessian references.
    ObjectiveResult(double& f, VectorRef g, MatrixRef H) : f(f), g(g), H(H) {}
};

/// The result of the evaluation of an objective function in Python.
/// @see ObjectiveFunction4py
struct ObjectiveResult4py
{
    /// The evaluated value of the objective function.
    double f = 0.0;

    /// The evaluated gradient of the objective function.
    VectorRef g;

    /// The evaluated Hessian of the objective function.
    MatrixRef4py H;

    /// The requirements in the evaluation of the objective function.
    ObjectiveRequirement requires;

    /// The boolean flag that indicates if the objective function evaluation failed.
    bool failed = false;

    /// Construct an ObjectiveResult4py object with given ObjectiveResult object.
    ObjectiveResult4py(ObjectiveResult& res)
    : g(res.g), H(res.H), requires(res.requires) {}
};

/// The functional signature of an objective function.
/// @param x The values of the variables \eq{x}.
/// @return An ObjectiveResult object with the evaluated result of the objective function.
using ObjectiveFunction = std::function<void(VectorConstRef, ObjectiveResult&)>;

/// The functional signature of an objective function in Python.
/// @param x The values of the variables \eq{x}.
/// @return An ObjectiveResult4py object with the evaluated result of the objective function.
using ObjectiveFunction4py = std::function<void(VectorConstRef, ObjectiveResult4py*)>;

/// Convert an ObjectiveFunction4Py function to an ObjectiveFunction function.
inline auto convert(const ObjectiveFunction4py& obj4py)
{
    return [=](VectorConstRef x, ObjectiveResult& res)
    {
        ObjectiveResult4py res4py(res);
        obj4py(x, &res4py);
        res.f = res4py.f;
    };
}

} // namespace Optima
