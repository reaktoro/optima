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

    /// The coefficient matrix \eq{A_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    MatrixRef Ae;

    /// The coefficient matrix \eq{A_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    MatrixRef Ag;

    /// The right-hand side vector \eq{b_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    VectorRef be;

    /// The right-hand side vector \eq{b_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    VectorRef bg;

    /// The nonlinear equality constraint function in \eq{h_{\mathrm{e}}(x)=0}.
    ConstraintFunction he;

    /// The nonlinear inequality constraint function in \eq{h_{\mathrm{g}}(x)\geq0}.
    ConstraintFunction hg;

    /// The objective function \eq{f(x)} of the optimization problem.
    ObjectiveFunction f;

    /// The lower bounds of the variables \eq{x}.
    VectorRef xlower;

    /// The upper bounds of the variables \eq{x}.
    VectorRef xupper;

    /// The derivatives *∂g/∂p*.
    Matrix dgdp;

    /// The derivatives *∂h/∂p*.
    Matrix dhdp;

    /// The derivatives *∂b/∂p*.
    Matrix dbdp;

    /// The nonlinear equality constraint function in \eq{h_{\mathrm{e}}(x)=0} (to be used in python only!).
    ConstraintFunction4py __4py_he;

    /// The nonlinear inequality constraint function in \eq{h_{\mathrm{g}}(x)\geq0} (to be used in python only!).
    ConstraintFunction4py __4py_hg;

    /// The objective function \eq{f(x)} of the optimization problem (to be used in python only!).
    ObjectiveFunction4py __4py_f;
};

} // namespace Optima
