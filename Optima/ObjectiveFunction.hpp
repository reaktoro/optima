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

/// The result of the evaluation of an objective function.
/// @see ObjectiveFunction
struct ObjectiveResult
{
    double& f;     ///< The evaluated objective function *f(x, p)*.
    VectorRef fx;  ///< The evaluated gradient of *f(x, p)* with respect to *x*.
    MatrixRef fxx; ///< The evaluated Jacobian *fx(x, p)* with respect to *x*.
    MatrixRef fxp; ///< The evaluated Jacobian *fx(x, p)* with respect to *p*.
    bool& diagfxx; ///< The flag indicating whether `fxx` is diagonal.
};

/// The result of the evaluation of an objective function in Python.
/// @see ObjectiveFunction4py
struct ObjectiveResult4py
{
    double f;         ///< The evaluated objective function *f(x, p)*.
    VectorRef fx;     ///< The evaluated gradient of *f(x, p)* with respect to *x*.
    MatrixRef4py fxx; ///< The evaluated Jacobian *fx(x, p)* with respect to *x*.
    MatrixRef4py fxp; ///< The evaluated Jacobian *fx(x, p)* with respect to *p*.
    bool diagfxx;     ///< The flag indicating whether `fxx` is diagonal.
};

/// The functional signature of an objective function.
/// @param x The values of the primal variables *x*.
/// @param p The values of the parameter variables *p*.
/// @param res The evaluated objective function and its first and second order derivatives.
/// @return Return `true` if the evaluation succeeded, `false` otherwise.
using ObjectiveFunction = std::function<bool(VectorConstRef x, VectorConstRef p, ObjectiveResult res)>;

/// The functional signature of an objective function in Python.
/// @param x The values of the primal variables *x*.
/// @param p The values of the parameter variables *p*.
/// @param res The evaluated objective function and its first and second order derivatives.
/// @return Return `true` if the evaluation succeeded, `false` otherwise.
using ObjectiveFunction4py = std::function<bool(VectorConstRef, VectorConstRef, ObjectiveResult4py*)>;

/// Convert an ObjectiveFunction4py function to an ObjectiveFunction function.
inline auto convert(const ObjectiveFunction4py& obj4py) -> ObjectiveFunction
{
    return [=](VectorConstRef x, VectorConstRef p, ObjectiveResult res)
    {
        ObjectiveResult4py res4py{res.f, res.fx, res.fxx, res.fxp, false};
        const auto status = obj4py(x, p, &res4py);
        res.f = res4py.f;
        res.diagfxx = res4py.diagfxx;
        return status;
    };
}

} // namespace Optima
