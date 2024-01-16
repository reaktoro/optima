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

// C++ includes
#include <functional>

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// The result of a constraint function evaluation.
/// @see ConstraintFunction
template<typename Bool, typename Vec, typename Mat>
struct ConstraintResultBase
{
    /// The evaluated vector value of *q(x, p, c)*.
    Vec val;

    /// The evaluated Jacobian matrix of *q(x, p, c)* with respect to *x*.
    Mat ddx;

    /// The evaluated Jacobian matrix of *q(x, p, c)* with respect to *p*.
    Mat ddp;

    /// The evaluated Jacobian matrix of *q(x, p, c)* with respect to *c*.
    Mat ddc;

    /// True if `ddx` is non-zero only on columns corresponding to basic varibles in *x*.
    Bool ddx4basicvars;

    /// True if the constraint function evaluation succeeded.
    Bool succeeded;

    /// Construct a default ConstraintResultBase object.
    ConstraintResultBase()
    : ConstraintResultBase(0, 0, 0, 0) {}

    /// Construct a ConstraintResultBase object with given dimensions.
    /// @param nq The number of constraint equations in *q(x, p, c)*.
    /// @param nx The number of variables in *x*.
    /// @param np The number of parameters in *p*.
    /// @param nc The number of sensitive parameters in *c*.
    ConstraintResultBase(Index nq, Index nx, Index np, Index nc)
    : val(nq), ddx(nq, nx), ddp(nq, np), ddc(nq, nc), ddx4basicvars(false), succeeded(true) {}

    /// Construct an ConstraintResultBase object from another.
    template<typename B, typename V, typename M>
    ConstraintResultBase(ConstraintResultBase<B, V, M>& other)
    : val(other.val), ddx(other.ddx), ddp(other.ddp), ddc(other.ddc),
      ddx4basicvars(other.ddx4basicvars), succeeded(other.succeeded) {}

    /// Construct an ConstraintResultBase object with given data.
    ConstraintResultBase(Vec val, Mat ddx, Mat ddp, Mat ddc, Bool ddx4basicvars, Bool succeeded)
    : val(val), ddx(ddx), ddp(ddp), ddc(ddc), ddx4basicvars(ddx4basicvars), succeeded(succeeded) {}

    /// Resize this ConstraintResultBase object with given dimensions.
    /// @param nq The number of constraint equations in *q(x, p, c)*.
    /// @param nx The number of variables in *x*.
    /// @param np The number of parameters in *p*.
    /// @param nc The number of sensitive parameters in *c*.
    auto resize(Index nq, Index nx, Index np, Index nc) -> void
    {
        val.resize(nq);
        ddx.resize(nq, nx);
        ddp.resize(nq, np);
        ddc.resize(nq, nc);
    }
};

/// The result of a constraint function evaluation.
using ConstraintResult = ConstraintResultBase<bool, Vector, Matrix>;

/// The result of a constraint function evaluation.
using ConstraintResultRef = ConstraintResultBase<bool&, VectorRef, MatrixRef>;

/// The options transmitted to the evaluation of a constraint function.
struct ConstraintOptions
{
    /// Used to list the constraint function components that need to be evaluated.
    struct Eval
    {
        bool ddx = true;  ///< True if evaluating the Jacobian matrix of *q(x, p, c)* with respect to *x* is needed.
        bool ddp = true;  ///< True if evaluating the Jacobian matrix of *q(x, p, c)* with respect to *p* is needed.
        bool ddc = false; ///< True if evaluating the Jacobian matrix of *q(x, p, c)* with respect to *p* is needed.
    };

    /// The constraint function components that need to be evaluated.
    const Eval eval;

    /// The indices of the basic variables in *x*.
    IndicesView ibasicvars;
};

/// Used to represent a constraint function *q(x, p, c)*.
class ConstraintFunction
{
public:
    /// The main functional signature of a constraint function *q(x, p, c)*.
    /// @param[out] res The evaluated result of the constraint function and its derivatives.
    /// @param x The primal variables *x*.
    /// @param p The parameter variables *p*.
    /// @param c The sensitive parameter variables *c*.
    /// @param opts The options transmitted to the evaluation of *q(x, p, c)*.
    using Signature = std::function<void(ConstraintResultRef res, VectorView x, VectorView p, VectorView c, ConstraintOptions opts)>;

    /// The functional signature of a constraint function *q(x, p, c)* incoming from Python.
    /// @param[out] res The evaluated result of the constraint function and its derivatives.
    /// @param x The primal variables *x*.
    /// @param p The parameter variables *p*.
    /// @param c The sensitive parameter variables *c*.
    /// @param opts The options transmitted to the evaluation of *q(x, p, c)*.
    using Signature4py = std::function<void(ConstraintResultRef* res, VectorView x, VectorView p, VectorView c, ConstraintOptions opts)>;

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
    auto operator()(ConstraintResultRef res, VectorView x, VectorView p, VectorView c, ConstraintOptions opts) const -> void;

    /// Assign another constraint function to this.
    auto operator=(const Signature& fn) -> ConstraintFunction&;

    /// Return `true` if this ConstraintFunction object has been initialized.
    auto initialized() const -> bool;

private:
    /// The constraint function with main functional signature.
    Signature fn;
};


} // namespace Optima
