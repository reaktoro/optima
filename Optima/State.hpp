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
#include <memory>

// Optima includes
#include <Optima/Dims.hpp>
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/Stability.hpp>

namespace Optima {

/// The state of the optimization variables.
class State
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a State object with given dimensions information.
    State(const Dims& dims);

    /// Construct a copy of a State instance.
    State(const State& other);

    /// Destroy this State instance.
    virtual ~State();

    /// Assign a State instance to this.
    auto operator=(State other) -> State& = delete;

    /// The dimension information of variables and constraints in the optimization problem.
    Dims const dims;

    /// The variables \eq{x} of the optimization problem.
    VectorRef x;

    /// The parameter variables \eq{p} of the optimization problem.
    VectorRef p;

    /// The Lagrange multipliers \eq{y=(y_{\mathrm{e}},y_{\mathrm{g}})} of the optimization problem.
    VectorRef y;

    /// The Lagrange multipliers with respect to constraints \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    VectorRef ye;

    /// The Lagrange multipliers with respect to constraints \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\geq b_{\mathrm{g}}}.
    VectorRef yg;

    /// The Lagrange multipliers \eq{z=(z_{\mathrm{e}},z_{\mathrm{g}})} of the optimization problem.
    VectorRef z;

    /// The Lagrange multipliers with respect to constraints \eq{h_{\mathrm{e}}(x)=0}.
    VectorRef ze;

    /// The Lagrange multipliers with respect to constraints \eq{h_{\mathrm{g}}(x)\geq0}.
    VectorRef zg;

    /// The stability measures of variables \eq{x} defined as \eq{s=g+A_{\mathrm{ex}}^{T}y_{\mathrm{e}}+A_{\mathrm{gx}}^{T}y_{\mathrm{g}}+J_{\mathrm{ex}}^{T}z_{\mathrm{e}}+J_{\mathrm{gx}}^{T}z_{\mathrm{g}}}.
    VectorRef s;

    /// The sensitivity derivatives \eq{\partial x/\partial w} with respect to parameters \eq{w}.
    Matrix dxdw;

    /// The sensitivity derivatives \eq{\partial p/\partial w} with respect to parameters \eq{w}.
    Matrix dpdw;

    /// The sensitivity derivatives \eq{\partial y/\partial w} with respect to parameters \eq{w}.
    Matrix dydw;

    /// The sensitivity derivatives \eq{\partial z/\partial w} with respect to parameters \eq{w}.
    Matrix dzdw;

    /// The sensitivity derivatives \eq{\partial s/\partial w} with respect to parameters \eq{w}.
    Matrix dsdw;

    /// The variables \eq{(x,x_{\mathrm{b_{g}}},x_{\mathrm{h_{g}}})} of the basic optimization problem.
    VectorRef xbar;

    /// The variables \eq{(s,y_{\mathrm{g}},z_{\mathrm{g}})} of the basic optimization problem.
    VectorRef sbar;

    /// The variables \eq{x_{b_{\mathrm{g}}}} in \eq{(x,x_{\mathrm{b_{g}}},x_{\mathrm{h_{g}}})} of the basic optimization problem.
    VectorRef xbg;

    /// The variables \eq{x_{h_{\mathrm{g}}}} in \eq{(x,x_{\mathrm{b_{g}}},x_{\mathrm{h_{g}}})} of the basic optimization problem.
    VectorRef xhg;

    /// The stability state of the primal variables *x*.
    Stability stability;
};

} // namespace Optima
