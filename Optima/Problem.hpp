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
#include <memory>

// Optima includes
#include <Optima/ConstraintFunction.hpp>
#include <Optima/Dims.hpp>
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/ObjectiveFunction.hpp>
#include <Optima/ResourcesFunction.hpp>

namespace Optima {

/// The class used to define an optimization problem.
class Problem
{
public:
    Dims const dims;       ///< The dimensions of the variables and constraints in the optimization problem.
    ResourcesFunction r;   ///< The optional function that precomputes shared resources for objective and constraint functions.
    ObjectiveFunction f;   ///< The objective function \eq{f(x, p)} of the optimization problem.
    ConstraintFunction he; ///< The nonlinear equality constraint function \eq{h_{\mathrm{e}}(x, p)=0}.
    ConstraintFunction hg; ///< The nonlinear inequality constraint function \eq{h_{\mathrm{g}}(x, p)\geq0}.
    ConstraintFunction v;  ///< The external nonlinear constraint function \eq{v(x, p)=0}.
    FixedMatrix Aex;       ///< The coefficient matrix \eq{A_{\mathrm{ex}}} in the linear equality constraints \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    FixedMatrix Aep;       ///< The coefficient matrix \eq{A_{\mathrm{ep}}} in the linear equality constraints \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    FixedMatrix Agx;       ///< The coefficient matrix \eq{A_{\mathrm{gx}}} in the linear inequality constraints \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\geq b_{\mathrm{g}}}.
    FixedMatrix Agp;       ///< The coefficient matrix \eq{A_{\mathrm{gp}}} in the linear inequality constraints \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\geq b_{\mathrm{g}}}.
    FixedVector be;        ///< The right-hand side vector \eq{b_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    FixedVector bg;        ///< The right-hand side vector \eq{b_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\geq b_{\mathrm{g}}}.
    FixedVector xlower;    ///< The lower bounds of the primal variables \eq{x}.
    FixedVector xupper;    ///< The upper bounds of the primal variables \eq{x}.
    FixedVector plower;    ///< The lower bounds of the parameter variables \eq{p}.
    FixedVector pupper;    ///< The upper bounds of the parameter variables \eq{p}.
    FixedVector c;         ///< The sensitivity parameters *c*.
    FixedMatrix bec;       ///< The Jacobian matrix \eq{\partial b_{\mathrm{e}}/\partial c} for calculation of sensitivity derivatives with respect to *c*.
    FixedMatrix bgc;       ///< The Jacobian matrix \eq{\partial b_{\mathrm{g}}/\partial c} for calculation of sensitivity derivatives with respect to *c*.

    /// Construct a default Problem instance.
    Problem();

    /// Construct a Problem instance with given dimensions.
    explicit Problem(const Dims& dims);

    /// Assign a Problem instance to this.
    auto operator=(const Problem& other) -> Problem&;
};

} // namespace Optima
