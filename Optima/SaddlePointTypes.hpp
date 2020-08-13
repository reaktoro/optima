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

/// The dimension details in a general saddle point problem.
struct SaddlePointProblemDims
{
    Index n   = 0; ///< The number of variables in *x* and *p*.
    Index nx  = 0; ///< The number of variables in *x = (xs, xu)*.
    Index np  = 0; ///< The number of variables in *p*.
    Index ns  = 0; ///< The number of variables in *xs*.
    Index nu  = 0; ///< The number of variables in *xu*.
    Index m   = 0; ///< The number of variables *y = (yl, yn)*.
    Index ml  = 0; ///< The number of variables *yl* (i.e., number of rows in matrix *A*).
    Index mn  = 0; ///< The number of variables *yn* (i.e., number of rows in matrix *J*).
    Index nb  = 0; ///< The number of basic variables in *x*.
    Index nn  = 0; ///< The number of non-basic variables *x*.
    Index nl  = 0; ///< The number of linearly dependent rows in *W = [A; J]*
    Index nbs = 0; ///< The number of basic variables in *xs*.
    Index nbu = 0; ///< The number of basic variables in *xu*.
    Index nns = 0; ///< The number of non-basic variables in *xs*.
    Index nnu = 0; ///< The number of non-basic variables in *xu*.
    Index nbe = 0; ///< The number of pivot/explicit basic variables in *xs*.
    Index nne = 0; ///< The number of pivot/explicit non-basic variables in *xs*.
    Index nbi = 0; ///< The number of non-pivot/implicit basic variables in *xs*.
    Index nni = 0; ///< The number of non-pivot/implicit non-basic variables in *xs*.
};

/// The coefficient matrix in a general saddle point problem.
struct SaddlePointMatrix
{
    MatrixConstRef H;   ///< The matrix block *H* in the saddle point matrix.
    MatrixConstRef A;   ///< The upper and constant block *A* of matrix *W = [A; J]* in the saddle point problem.
    MatrixConstRef J;   ///< The lower and variable block *J* of matrix *W = [A; J]* in the saddle point matrix.
    IndicesConstRef ju; ///< The indices of the *xu* variables in *x = (xs, xu, xp)*.
    IndicesConstRef jp; ///< The indices of the *xp* variables in *x = (xs, xu, xp)*.
};

/// The coefficient matrix in a canonical saddle point problem.
struct CanonicalSaddlePointMatrix
{
    SaddlePointProblemDims dims; ///< The dimension details in a general saddle point problem.
    MatrixConstRef Hss;          ///< The Hessian matrix block `Hss` in the canonical saddle point problem.
    MatrixConstRef Hsp;          ///< The Hessian matrix block `Hsp` in the canonical saddle point problem.
    MatrixConstRef Hps;          ///< The Hessian matrix block `Hps` in the canonical saddle point problem.
    MatrixConstRef Hpp;          ///< The Hessian matrix block `Hpp` in the canonical saddle point problem.
    MatrixConstRef Sbsns;        ///< The matrix block `Sbsns` in the canonical saddle point problem.
    MatrixConstRef Sbsnp;        ///< The matrix block `Sbsnp` in the canonical saddle point problem.
};

/// The representation of a canonical saddle point problem.
struct CanonicalSaddlePointProblem : CanonicalSaddlePointMatrix
{
    VectorConstRef as;  ///< The right-hand side vector `as` for the xs variables.
    VectorConstRef ap;  ///< The right-hand side vector `ap` for the xp variables.
    VectorConstRef bbs; ///< The right-hand side vector `bbs`.
    VectorRef xs;       ///< The solution vector `xs` in the canonical saddle point problem.
    VectorRef p;        ///< The solution vector `p` in the canonical saddle point problem.
    VectorRef ybs;      ///< The solution vector `ybs` in the canonical saddle point problem.
};

} // namespace Optima
