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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// The result of an objective function evaluation.
/// @see ObjectiveFunction
struct ObjectiveResult
{
    /// The evaluated objective function *f(x, p)*.
    double f = 0.0;

    /// The evaluated gradient vector of *f(x, p)* with respect to *x*.
    FixedVector fx;

    /// The evaluated Jacobian matrix of *fx(x, p)* with respect to *x*.
    FixedMatrix fxx;

    /// The evaluated Jacobian matrix of *fx(x, p)* with respect to *p*.
    FixedMatrix fxp;

    /// True if `fxx` is diagonal.
    bool diagfxx = false;

    /// True if `fxx` is non-zero only on columns corresponding to basic varibles in *x*.
    bool fxx4basicvars = false;

    /// Construct an ObjectiveResult object.
    /// @param nx The number of variables in *x*.
    /// @param np The number of variables in *p*.
    ObjectiveResult(Index nx, Index np)
    : fx(nx), fxx(nx, nx), fxp(nx, np) {}
};

/// The options transmitted to the evaluation of an objective function.
/// @see ObjectiveFunction, ObjectiveResult
struct ObjectiveOptions
{
    /// Used to list the objective function components that need to be evaluated.
    struct Eval
    {
        bool fxx = true; ///< True if evaluating the Jacobian matrix *fxx* is needed.
        bool fxp = true; ///< True if evaluating the Jacobian matrix *fxp* is needed.
    };

    /// The objective function components that need to be evaluated.
    const Eval eval;

    /// The indices of the basic variables in *x*.
    IndicesConstRef ibasicvars;
};

/// The functional signature of an objective function *f(x, p)*.
/// @param[out] res The evaluated result of the objective function and its derivatives.
/// @param x The primal variables *x*.
/// @param p The parameter variables *p*.
/// @param opts The options transmitted to the evaluation of *f(x, p)*.
/// @return Return `true` if the evaluation succeeded, `false` otherwise.
/// @see ObjectiveResult, ObjectiveOptions
using ObjectiveFunction = std::function<bool(ObjectiveResult& res, VectorConstRef x, VectorConstRef p, ObjectiveOptions opts)>;

} // namespace Optima
