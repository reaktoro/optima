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

/// The result of the evaluation of a constraint function.
struct ConstraintResult
{
    VectorRef h;  ///< The evaluated equality constraint function *h(x, p)*.
    MatrixRef hx; ///< The evaluated Jacobian of *h(x, p)* with respect to *x*.
    MatrixRef hp; ///< The evaluated Jacobian of *h(x, p)* with respect to *p*.
};

/// The result of the evaluation of a constraint function in Python.
struct ConstraintResult4py
{
    VectorRef h;     ///< The evaluated equality constraint function *h(x, p)*.
    MatrixRef4py hx; ///< The evaluated Jacobian of *h(x, p)* with respect to *x*.
    MatrixRef4py hp; ///< The evaluated Jacobian of *h(x, p)* with respect to *p*.
};

/// The signature of a constraint function.
using ConstraintFunction = std::function<bool(VectorConstRef, VectorConstRef, ConstraintResult)>;

/// The signature of a constraint function in Python.
using ConstraintFunction4py = std::function<bool(VectorConstRef, VectorConstRef, ConstraintResult4py*)>;

/// Convert an ConstraintFunction4py function to a ConstraintFunction function.
inline auto convert(const ConstraintFunction4py& obj4py) -> ConstraintFunction
{
    return [=](VectorConstRef x, VectorConstRef p, ConstraintResult res)
    {
        ConstraintResult4py res4py({res.h, res.hx, res.hp});
        return obj4py(x, p, &res4py);
    };
}

} // namespace Optima
