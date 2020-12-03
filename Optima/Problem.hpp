// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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

namespace Optima {

/// The class used to define an optimization problem.
class Problem
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a Problem instance with given dimension information.
    Problem(const Dims& dims);

    /// Construct a copy of a Problem instance.
    Problem(const Problem& other);

    /// Destroy this Problem instance.
    virtual ~Problem();

    /// Assign a Problem instance to this.
    auto operator=(Problem other) -> Problem& = delete;

    /// The dimension information of variables and constraints in the optimization problem.
    Dims const dims;

    /// The coefficient matrix \eq{A_{\mathrm{ex}}} in the linear equality constraints \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    MatrixRef Aex;

    /// The coefficient matrix \eq{A_{\mathrm{ep}}} in the linear equality constraints \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    MatrixRef Aep;

    /// The coefficient matrix \eq{A_{\mathrm{gx}}} in the linear inequality constraints \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\geq b_{\mathrm{g}}}.
    MatrixRef Agx;

    /// The coefficient matrix \eq{A_{\mathrm{gp}}} in the linear inequality constraints \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\geq b_{\mathrm{g}}}.
    MatrixRef Agp;

    /// The right-hand side vector \eq{b_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    VectorRef be;

    /// The right-hand side vector \eq{b_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\geq b_{\mathrm{g}}}.
    VectorRef bg;

    /// The nonlinear equality constraint function \eq{h_{\mathrm{e}}(x, p)=0}.
    ConstraintFunction he;

    /// The nonlinear inequality constraint function \eq{h_{\mathrm{g}}(x, p)\geq0}.
    ConstraintFunction hg;

    /// The external nonlinear constraint function \eq{v(x, p)=0}.
    ConstraintFunction v;

    /// The objective function \eq{f(x, p)} of the optimization problem.
    ObjectiveFunction f;

    /// The lower bounds of the primal variables \eq{x}.
    VectorRef xlower;

    /// The upper bounds of the primal variables \eq{x}.
    VectorRef xupper;

    /// The lower bounds of the parameter variables \eq{p}.
    VectorRef plower;

    /// The upper bounds of the parameter variables \eq{p}.
    VectorRef pupper;

    /// The derivatives *∂fx/∂w*.
    Matrix fxw;

    /// The derivatives *∂b/∂w*.
    Matrix bw;

    /// The derivatives *∂h/∂w*.
    Matrix hw;

    /// The derivatives *∂v/∂w*.
    Matrix vw;
};

} // namespace Optima
