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

/// The result of an objective function evaluation.
/// @see ObjectiveFunction
template<typename Real, typename Bool, typename Vec, typename Mat>
struct ObjectiveResultBase
{
    /// The evaluated objective function *f(x, p, c)*.
    Real f;

    /// The evaluated gradient vector of *f(x, p, c)* with respect to *x*.
    Vec fx;

    /// The evaluated Jacobian matrix of *fx(x, p, c)* with respect to *x*.
    Mat fxx;

    /// The evaluated Jacobian matrix of *fx(x, p, c)* with respect to *p*.
    Mat fxp;

    /// The evaluated Jacobian matrix of *fx(x, p, c)* with respect to sensitive parameters *c*.
    Mat fxc;

    /// True if `fxx` is diagonal.
    Bool diagfxx;

    /// True if `fxx` is non-zero only on columns corresponding to basic varibles in *x*.
    Bool fxx4basicvars;

    /// True if the objective function evaluation succeeded.
    Bool succeeded;

    /// Construct a default ObjectiveResultBase object.
    ObjectiveResultBase()
    : ObjectiveResultBase(0, 0, 0) {}

    /// Construct an ObjectiveResultBase object with given dimensions.
    /// @param nx The number of variables in *x*.
    /// @param np The number of parameters in *p*.
    /// @param nc The number of sensitive parameters in *c*.
    ObjectiveResultBase(Index nx, Index np, Index nc)
    : f(0.0), fx(nx), fxx(nx, nx), fxp(nx, np), fxc(nx, nc),
      diagfxx(false), fxx4basicvars(false), succeeded(true) {}

    /// Construct an ObjectiveResultBase object from another.
    template<typename R, typename B, typename V, typename M>
    ObjectiveResultBase(ObjectiveResultBase<R, B, V, M>& other)
    : f(other.f), fx(other.fx), fxx(other.fxx), fxp(other.fxp), fxc(other.fxc),
      diagfxx(other.diagfxx), fxx4basicvars(other.fxx4basicvars),
      succeeded(other.succeeded) {}

    /// Construct an ObjectiveResultBase object with given data.
    ObjectiveResultBase(Real f, Vec fx, Mat fxx, Mat fxp, Mat fxc, Bool diagfxx, Bool fxx4basicvars, Bool succeeded)
    : f(f), fx(fx), fxx(fxx), fxp(fxp), fxc(fxc), diagfxx(diagfxx),
      fxx4basicvars(fxx4basicvars), succeeded(succeeded) {}

    /// Resize this ObjectiveResultBase object with given dimensions.
    /// @param nx The number of variables in *x*.
    /// @param np The number of parameters in *p*.
    /// @param nc The number of sensitive parameters in *c*.
    auto resize(Index nx, Index np, Index nc) -> void
    {
        fx.resize(nx);
        fxx.resize(nx, nx);
        fxp.resize(nx, np);
        fxc.resize(nx, nc);
    }
};

/// The result of an objective function evaluation.
using ObjectiveResult = ObjectiveResultBase<double, bool, Vector, Matrix>;

/// The result of an objective function evaluation.
using ObjectiveResultRef = ObjectiveResultBase<double&, bool&, VectorRef, MatrixRef>;

/// The options transmitted to the evaluation of an objective function.
/// @see ObjectiveFunction, ObjectiveResult
struct ObjectiveOptions
{
    /// Used to list the objective function components that need to be evaluated.
    struct Eval
    {
        bool fxx = true;  ///< True if evaluating the Jacobian matrix *fxx* is needed.
        bool fxp = true;  ///< True if evaluating the Jacobian matrix *fxp* is needed.
        bool fxc = false; ///< True if evaluating the Jacobian matrix *fxc* is needed.
    };

    /// The objective function components that need to be evaluated.
    const Eval eval;

    /// The indices of the basic variables in *x*.
    IndicesView ibasicvars;
};

/// Used to represent an objective function *f(x, p, c)*.
/// @see ConstraintFunction
class ObjectiveFunction
{
public:
    /// The main functional signature of an objective function *f(x, p, c)*.
    /// @param[out] res The evaluated result of the objective function and its derivatives.
    /// @param x The primal variables *x*.
    /// @param p The parameter variables *p*.
    /// @param c The sensitive parameter variables *c*.
    /// @param opts The options transmitted to the evaluation of *f(x, p, c)*.
    using Signature = std::function<void(ObjectiveResultRef res, VectorView x, VectorView p, VectorView c, ObjectiveOptions opts)>;

    /// The functional signature of an objective function *f(x, p, c)* incoming from Python.
    /// @param[out] res The evaluated result of the objective function and its derivatives.
    /// @param x The primal variables *x*.
    /// @param p The parameter variables *p*.
    /// @param c The sensitive parameter variables *c*.
    /// @param opts The options transmitted to the evaluation of *f(x, p, c)*.
    using Signature4py = std::function<void(ObjectiveResultRef* res, VectorView x, VectorView p, VectorView c, ObjectiveOptions opts)>;

    /// Construct a default ObjectiveFunction object.
    ObjectiveFunction();

    /// Construct an ObjectiveFunction object with given function.
    ObjectiveFunction(const Signature& fn);

    /// Construct an ObjectiveFunction object with given function.
    ObjectiveFunction(const Signature4py& fn);

    // template<typename Func, EnableIf<!isStdFunction<Func>>...>
    // ObjectiveFunction(const Func& fn)
    // : ObjectiveFunction(fn) {}

    /// Evaluate the objective function.
    auto operator()(ObjectiveResultRef res, VectorView x, VectorView p, VectorView c, ObjectiveOptions opts) const -> void;

    /// Assign another objective function to this.
    auto operator=(const Signature& fn) -> ObjectiveFunction&;

    /// Return `true` if this ObjectiveFunction object has been initialized.
    auto initialized() const -> bool;

private:
    /// The objective function with main functional signature.
    Signature fn;
};

} // namespace Optima
