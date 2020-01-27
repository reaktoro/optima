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
    /// The value of the evaluated constraint function.
    VectorRef h;

    /// The Jacobian matrix of the evaluated constraint function.
    MatrixRef J;

    /// The boolean flag that indicates if the constraint function evaluation failed.
    bool failed = false;
};

/// The result of the evaluation of a constraint function in Python.
struct ConstraintResult4py
{
    /// The value of the evaluated constraint function.
    VectorRef h;

    /// The Jacobian matrix of the evaluated constraint function.
    MatrixRef4py J;

    /// The boolean flag that indicates if the constraint function evaluation failed.
    bool failed = false;

    /// Construct a ConstraintResult4py object with given ConstraintResult object.
    ConstraintResult4py(ConstraintResult& res) : h(res.h), J(res.J) {}
};

/// The signature of a constraint function.
using ConstraintFunction = std::function<void(VectorConstRef, ConstraintResult&)>;

/// The signature of a constraint function in Python.
using ConstraintFunction4py = std::function<void(VectorConstRef, ConstraintResult4py*)>;

/// Convert an ConstraintFunction4Py function to a ConstraintFunction function.
inline auto convert(const ConstraintFunction4py& obj4py)
{
    return [=](VectorConstRef x, ConstraintResult& res)
    {
        ConstraintResult4py res4py(res);
        obj4py(x, &res4py);
    };
}

} // namespace Optima
