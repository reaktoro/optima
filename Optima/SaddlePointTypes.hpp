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

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// The dimension details in a saddle point problem.
struct SaddlePointProblemDims
{
    Index n   = 0; ///< The number of variables *x* and columns in *W = [A; J]*.
    Index m   = 0; ///< The number of variables *y* and rows in *W = [A; J]*.
    Index ml  = 0; ///< The number of rows in matrix *A*.
    Index mn  = 0; ///< The number of rows in matrix *J*.
    Index nb  = 0; ///< The number of basic variables in *x*.
    Index nn  = 0; ///< The number of non-basic variables *x*.
    Index nl  = 0; ///< The number of linearly dependent rows in *W = [A; J]*
    Index nx  = 0; ///< The number of *free variables* in *x*.
    Index nf  = 0; ///< The number of *fixed variables* in *x*.
    Index nbx = 0; ///< The number of *free basic variables* in *x*.
    Index nbf = 0; ///< The number of *fixed basic variables* in *x*.
    Index nnx = 0; ///< The number of *free non-basic variables* in *x*.
    Index nnf = 0; ///< The number of *fixed non-basic variables* in *x*.
    Index nbe = 0; ///< The number of *pivot free basic variables* in *x*.
    Index nne = 0; ///< The number of *pivot free non-basic variables* in *x*.
    Index nbi = 0; ///< The number of *non-pivot free basic variables* in *x*.
    Index nni = 0; ///< The number of *non-pivot free non-basic variables* in *x*.
};

/// The coefficient matrix in a canonical saddle point problem.
struct CanonicalSaddlePointMatrix
{
    SaddlePointProblemDims dims; ///< The dimension details in a saddle point problem.
    MatrixConstRef Hxx;          ///< The Hessian matrix block `Hxx` in the canonical saddle point problem.
    MatrixConstRef Sbxnx;        ///< The matrix block `Sbxnx` in the canonical saddle point problem.
};

/// The representation of a canonical saddle point problem.
struct CanonicalSaddlePointProblem : CanonicalSaddlePointMatrix
{
    VectorConstRef ax;  ///< The right-hand side vector `ax` for the free variables.
    VectorConstRef bbx; ///< The right-hand side vector `bbx`.
    VectorRef xx;       ///< The solution vector `xx` in the canonical saddle point problem.
    VectorRef ybx;      ///< The solution vector `ybx` in the canonical saddle point problem.
};

} // namespace Optima
