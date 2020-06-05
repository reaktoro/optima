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

    /// The Lagrange multipliers \eq{y=(y_{b_{\mathrm{e}}},y_{b_{\mathrm{g}}},y_{h_{\mathrm{e}}},y_{h_{\mathrm{g}}})} of the optimization problem.
    VectorRef y;

    /// The Lagrange multipliers with respect to constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    VectorRef ybe;

    /// The Lagrange multipliers with respect to constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    VectorRef ybg;

    /// The Lagrange multipliers with respect to constraints \eq{h_{\mathrm{e}}(x)=0}.
    VectorRef yhe;

    /// The Lagrange multipliers with respect to constraints \eq{h_{\mathrm{g}}(x)\geq0}.
    VectorRef yhg;

    /// The instability measures of variables \eq{x} defined as \eq{z=g+A_{\mathrm{e}}^{T}y_{b_{\mathrm{e}}}+A_{\mathrm{g}}^{T}y_{b_{\mathrm{g}}}+J_{\mathrm{e}}^{T}y_{h_{\mathrm{e}}}+J_{\mathrm{g}}^{T}y_{h_{\mathrm{g}}}}.
    VectorRef z;

    /// The sensitivity derivatives \eq{\partial x/\partial p} with respect to parameters \eq{p}.
    Matrix dxdp;

    /// The sensitivity derivatives \eq{\partial y/\partial p} with respect to parameters \eq{p}.
    Matrix dydp;

    /// The sensitivity derivatives \eq{\partial z/\partial p} with respect to parameters \eq{p}.
    Matrix dzdp;

    /// The variables \eq{\bar{x} = (x,x_{b_{\mathrm{g}}},x_{h_{\mathrm{g}}})} of the basic optimization problem.
    VectorRef xbar;

    /// The variables \eq{\bar{z}=(z,y_{b_{\mathrm{g}}},y_{h_{\mathrm{g}}})} of the basic optimization problem.
    VectorRef zbar;

    /// The variables \eq{x_{b_{\mathrm{g}}}} in \eq{\bar{x} = (x,x_{b_{\mathrm{g}}},x_{h_{\mathrm{g}}})} of the basic optimization problem.
    VectorRef xbg;

    /// The variables \eq{x_{h_{\mathrm{g}}}} in \eq{\bar{x} = (x,x_{b_{\mathrm{g}}},x_{h_{\mathrm{g}}})} of the basic optimization problem.
    VectorRef xhg;

    /// The stability state of the primal variables *x*.
    Stability stability;
};

} // namespace Optima
