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

/// The result of a constraint function evaluation.
/// @see ConstraintFunction
struct ConstraintResult
{
    /// The evaluated vector value of *c(x, p)*.
    FixedVector val;

    /// The evaluated Jacobian matrix of *c(x, p)* with respect to *x*.
    FixedMatrix ddx;

    /// The evaluated Jacobian matrix of *c(x, p)* with respect to *p*.
    FixedMatrix ddp;

    /// True if `ddx` is non-zero only on columns corresponding to basic varibles in *x*.
    bool ddx4basicvars = false;

    /// True if the constraint function evaluation succeeded.
    bool succeeded = true;

    /// Construct a ConstraintResult object.
    /// @param nc The number of constraint equations in *c(x, p)*.
    /// @param nx The number of variables in *x*.
    /// @param np The number of variables in *p*.
    ConstraintResult(Index nc, Index nx, Index np)
    : val(nc), ddx(nc, nx), ddp(nc, np) {}
};

/// The options transmitted to the evaluation of a constraint function.
struct ConstraintOptions
{
    /// Used to list the constraint function components that need to be evaluated.
    struct Eval
    {
        bool ddx = true; ///< True if evaluating the Jacobian matrix of *c(x, p)* with respect to *x* is needed.
        bool ddp = true; ///< True if evaluating the Jacobian matrix of *c(x, p)* with respect to *p* is needed.
    };

    /// The constraint function components that need to be evaluated.
    const Eval eval;

    /// The indices of the basic variables in *x*.
    IndicesView ibasicvars;
};

/// Used to represent a constraint function *c(x, p)*.
class ConstraintFunction
{
public:
    /// The main functional signature of a constraint function *c(x, p)*.
    /// @param[out] res The evaluated result of the constraint function and its derivatives.
    /// @param x The primal variables *x*.
    /// @param p The parameter variables *p*.
    /// @param opts The options transmitted to the evaluation of *c(x, p)*.
    using Signature = std::function<void(ConstraintResult& res, VectorView x, VectorView p, ConstraintOptions opts)>;

    /// The functional signature of a constraint function *c(x, p)* incoming from Python.
    /// @param[out] res The evaluated result of the constraint function and its derivatives.
    /// @param x The primal variables *x*.
    /// @param p The parameter variables *p*.
    /// @param opts The options transmitted to the evaluation of *c(x, p)*.
    using Signature4py = std::function<void(ConstraintResult* res, VectorView x, VectorView p, ConstraintOptions opts)>;

    /// Construct a default ConstraintFunction object.
    ConstraintFunction();

    /// Construct an ConstraintFunction object with given function.
    ConstraintFunction(const Signature& fn);

    /// Construct an ConstraintFunction object with given function.
    ConstraintFunction(const Signature4py& fn);

    // template<typename Func, EnableIf<!isStdFunction<Func>>...>
    // ConstraintFunction(const Func& fn)
    // : ConstraintFunction(fn) {}

    /// Evaluate the constraint function.
    auto operator()(ConstraintResult& res, VectorView x, VectorView p, ConstraintOptions opts) const -> void;

private:
    /// The constraint function with main functional signature.
    Signature fn;
};


} // namespace Optima
