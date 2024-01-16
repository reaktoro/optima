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
#include <Optima/Dims.hpp>
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// The state of the optimization variables.
class State
{
public:
    Dims const dims;     ///< The dimensions of the variables and constraints in the optimization problem.
    FixedVector x;       ///< The variables \eq{x} of the optimization problem.
    FixedVector p;       ///< The parameter variables \eq{p} of the optimization problem.
    FixedVector ye;      ///< The Lagrange multipliers \eq{y_{\mathrm{e}} with respect to constraints \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    FixedVector yg;      ///< The Lagrange multipliers \eq{y_{\mathrm{g}} with respect to constraints \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\geq b_{\mathrm{g}}}.
    FixedVector ze;      ///< The Lagrange multipliers \eq{z_{\mathrm{e}} with respect to constraints \eq{h_{\mathrm{e}}(x)=0}.
    FixedVector zg;      ///< The Lagrange multipliers \eq{z_{\mathrm{g}} with respect to constraints \eq{h_{\mathrm{g}}(x)\geq0}.
    FixedVector s;       ///< The stability measures of variables \eq{x} defined as \eq{s=g+A_{\mathrm{ex}}^{T}y_{\mathrm{e}}+A_{\mathrm{gx}}^{T}y_{\mathrm{g}}+J_{\mathrm{ex}}^{T}z_{\mathrm{e}}+J_{\mathrm{gx}}^{T}z_{\mathrm{g}}}.
    FixedVector xbg;     ///< The variables \eq{x_{b_{\mathrm{g}}}} in \eq{(x,x_{\mathrm{b_{g}}},x_{\mathrm{h_{g}}})} of the basic optimization problem.
    FixedVector xhg;     ///< The variables \eq{x_{h_{\mathrm{g}}}} in \eq{(x,x_{\mathrm{b_{g}}},x_{\mathrm{h_{g}}})} of the basic optimization problem.
    Indices js;          ///< The indices of the stable variables in *x*.
    Indices ju;          ///< The indices of the unstable variables in *x*.
    Indices jlu;         ///< The indices of the lower unstable variables in *x*.
    Indices juu;         ///< The indices of the upper unstable variables in *x*.
    Indices jb;          ///< The indices of the basic variables in *x*.
    Indices jn;          ///< The indices of the non-basic variables in *x*.

    /// Construct a default State object.
    State();

    /// Construct a State object with given dimensions.
    explicit State(const Dims& dims);

    /// Assign a State instance to this.
    auto operator=(const State& other) -> State&;
};

} // namespace Optima
