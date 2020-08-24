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
    bool f   = true; ///< The flag indicating if the objective function *f(x, p)* needs to be evaluated.
    bool fx  = true; ///< The flag indicating if the gradient function *fx(x, p)* needs to be evaluated.
    bool fxx = true; ///< The flag indicating if the Jacobian function *fxx(x, p)* needs to be evaluated.
    bool fxp = true; ///< The flag indicating if the Jacobian function *fxp(x, p)* needs to be evaluated.
};

/// The result of the evaluation of an objective function.
/// @see ObjectiveFunction
struct ObjectiveResult
{
    double& f;                     ///< The evaluated objective function *f(x, p)*.
    VectorRef fx;                  ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    MatrixRef fxx;                 ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *x*, i.e., the Hessian of *f(x, p)* with respect to *x*.
    MatrixRef fxp;                 ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *p*.
    ObjectiveRequirement requires; ///< The requirements in the evaluation of the objective function.
    bool failed = false;           ///< The boolean flag that indicates if the objective function evaluation failed.

    /// Construct an ObjectiveResult object.
    ObjectiveResult(double& f, VectorRef fx, MatrixRef fxx, MatrixRef fxp) : f(f), fx(fx), fxx(fxx), fxp(fxp) {}
};

/// The result of the evaluation of an objective function in Python.
/// @see ObjectiveFunction4py
struct ObjectiveResult4py
{
    double f;                      ///< The evaluated objective function *f(x, p)*.
    VectorRef fx;                  ///< The evaluated gradient of the objective function *f(x, p)* with respect to *x*.
    MatrixRef4py fxx;              ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *x*, i.e., the Hessian of *f(x, p)* with respect to *x*.
    MatrixRef4py fxp;              ///< The evaluated Jacobian of the gradient function *fx(x, p)* with respect to *p*.
    ObjectiveRequirement requires; ///< The requirements in the evaluation of the objective function.
    bool failed = false;           ///< The boolean flag that indicates if the objective function evaluation failed.

    /// Construct an ObjectiveResult4py object with given ObjectiveResult object.
    ObjectiveResult4py(ObjectiveResult& res) : fx(res.fx), fxx(res.fxx), fxp(res.fxp), requires(res.requires) {}
};

/// The functional signature of an objective function.
/// @param x The values of the primal variables \eq{x}.
/// @param p The values of the parameter variables \eq{p}.
/// @return An ObjectiveResult object with the evaluated result of the objective function.
using ObjectiveFunction = std::function<void(VectorConstRef, VectorConstRef, ObjectiveResult&)>;

/// The functional signature of an objective function in Python.
/// @param x The values of the primal variables \eq{x}.
/// @param p The values of the parameter variables \eq{p}.
/// @return An ObjectiveResult4py object with the evaluated result of the objective function.
using ObjectiveFunction4py = std::function<void(VectorConstRef, VectorConstRef, ObjectiveResult4py*)>;

/// Convert an ObjectiveFunction4py function to an ObjectiveFunction function.
inline auto convert(const ObjectiveFunction4py& obj4py)
{
    return [=](VectorConstRef x, VectorConstRef p, ObjectiveResult& res)
    {
        ObjectiveResult4py res4py(res);
        obj4py(x, p, &res4py);
        res.f = res4py.f;
    };
}

} // namespace Optima
