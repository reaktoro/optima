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
#include <Optima/Index.hpp>
#include <Optima/MatrixViewRWQ.hpp>
#include <Optima/MatrixViewW.hpp>

namespace Optima {

/// The arguments in method @ref Stability::update.
struct StabilityUpdateArgs
{
    MatrixView Wx;     ///< The matrix *Wx = [Ax; Jx]* in *W = [Ax Ap; Jx Jp]*.
    VectorView g;      ///< The gradient of the objective function with respect to x.
    VectorView x;      ///< The values of the variables x.
    VectorView w;      ///< The values of the variables w.
    VectorView xlower; ///< The lower bounds of the primal variables x.
    VectorView xupper; ///< The upper bounds of the primal variables x.
    IndicesView jb;    ///< The indices of the basic variables.
};

/// The stability status of the x variables.
/// This class computes the stability of each varible in \eq{x} using
/// \eqc{s = g - W_{\mathrm{x}}^{T}\lambda,}
/// where \eq{lambda} is defined as:
/// \eqc{\lambda = R^{T}g_{\mathrm{b}}.}
/// @see Stability::status
struct StabilityStatus
{
    IndicesView js;   ///< The indices of the stable variables in x.
    IndicesView ju;   ///< The indices of the unstable variables in x.
    IndicesView jlu;  ///< The indices of the lower unstable variables in x.
    IndicesView juu;  ///< The indices of the upper unstable variables in x.
    IndicesView jms;  ///< The indices of the meta-stable basic variables in x.
    VectorView s;     ///< The stability \eq{s=g-W_{\mathrm{x}}^{T}\lambda} of the x variables.
};

/// Used to determine the stability of the primal \eq{x} variables.
class Stability
{
private:
    Indices jsu;   ///< The indices of the x variables ordered as jsu = (js, jlu, juu).
    Index ns  = 0; ///< The number of stable variables in js.
    Index nbs = 0; ///< The number of stable basic variables in js.
    Index nlu = 0; ///< The number of lower unstable stable variables in jlu.
    Index nuu = 0; ///< The number of upper unstable stable variables in juu.
    Index nms = 0; ///< The number of meta-stable basic variables in jbs.
    Vector s;      ///< The stability \eq{s=g+W_{\mathrm{x}}^{T}w} of the *x* variables.

public:
    /// Construct a default Stability object.
    Stability();

    /// Construct a Stability object with given dimension.
    explicit Stability(Index nx);

    /// Update the stability status of the variables in x relative to a canonical form of matrix W.
    auto update(StabilityUpdateArgs args) -> void;

    /// Return the current stability status of the x variables.
    auto status() const -> StabilityStatus;
};

} // namespace Optima
