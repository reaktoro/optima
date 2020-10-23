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

namespace Optima {

/// Used to describe the possible methods for calculating Newton step.
enum class NewtonStepMethod
{
    /// This method solves the Newton step problem without any simplification.
    /// This method solves the Newton step problem by applying a partial-pivoting
    /// LU decomposition to the Jacobian matrix of dimension \eq{(n+m)\times(n+m)}.
    /// This method can be faster than the other methods for problems with small dimensions and
    /// when \eq{n} is not too larger than \eq{m}.
    /// @note This method takes no advantage of the particular structure of the Jacobian matrix.
    Fullspace,

    /// This method reduces the dimension of the Newton step problem from \eq{n+m} to \eq{n-m}.
    /// This method reduces the Newton step problem of dimension \eq{n+m} to an equivalent one of
    /// dimension \eq{n-m}, where \eq{n \times n} is the dimension of the Hessian matrix \eq{H} and
    /// \eq{m \times n} is the dimension of the Jacobian matrix \eq{A}.
    /// This method is suitable when matrix \eq{H} in the Newton step problem is dense and \eq{A}
    /// has relatively many rows to sufficiently decrease the size of the linear system.
    Nullspace,

    /// This method reduces the dimension of the Newton step problem from \eq{n+m} to \eq{m}.
    /// This method reduces the Newton step problem of dimension \eq{n+m} to an equivalent one of
    /// dimension \eq{m}, where these dimensions are related to the dimensions of the Hessian matrix
    /// \eq{H}, \eq{n \times n}, and Jacobian matrix \eq{A}, \eq{m \times n}.
    /// @warning This method should only be used when the Hessian matrix is a diagonal matrix.
    Rangespace,
};

/// Used to specify the options for the solution of Newton step problems.
/// @see NewtonStepSolver
struct NewtonStepOptions
{
    /// The method for solving the Newton step problems.
    NewtonStepMethod method = NewtonStepMethod::Rangespace;
};

} // namespace Optima
