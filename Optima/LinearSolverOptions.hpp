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

namespace Optima {

/// Used to describe the possible methods for solving the linear problems.
enum class LinearSolverMethod
{
    /// This method solves the linear problem without any simplification.
    /// This method solves the linear problem by applying a full-pivoting LU
    /// decomposition to the master matrix of dimension
    /// \eq{(n_x+n_p+n_w)\times(n_x+n_p+n_w)}. This method can be faster than
    /// the other methods for problems with small dimensions and when \eq{n_x}
    /// is not too larger than \eq{n_w}. @note This method takes no advantage
    /// of the particular structure of the master matrix.
    Fullspace,

    /// This method reduces the dimension of the linear problem from \eq{n_x+n_p+n_w} to \eq{n_x+n_p-n_w}.
    /// This method reduces the linear problem of dimension \eq{n_x+n_p+n_w} to
    /// an equivalent one of dimension \eq{n_x+n_p-n_w}, where \eq{n_x \times
    /// n_x} is the dimension of the Hessian matrix \eq{H_{xx}} and \eq{n_w
    /// \times n_x} is the dimension of the matrix \eq{W_x}. This method is
    /// suitable when matrix \eq{H_{xx}} in the linear problem is dense and \eq{W_x}
    /// has relatively many rows to sufficiently decrease the size of the
    /// linear system.
    Nullspace,

    /// This method reduces the dimension of the linear problem from \eq{n_x+n_p+n_w} to \eq{m}.
    /// This method reduces the linear problem of dimension \eq{n_x+n_p+n_w} to
    /// an equivalent one of dimension \eq{nw}, where these dimensions are
    /// related to the dimensions of the Hessian matrix \eq{H_{xx}}, \eq{n_x \times
    /// n_x}, and matrix \eq{W_x}, \eq{n_w \times n_x}.
    /// @warning This method should only be used when the Hessian matrix is diagonal.
    Rangespace,
};

/// Used to specify the options for the solution of linear problems.
/// @see LinearSolverSolver
struct LinearSolverOptions
{
    /// The method for solving the linear problems.
    LinearSolverMethod method = LinearSolverMethod::Nullspace;
};

} // namespace Optima
