// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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
#include <Optima/Core/SaddlePointMatrix.hpp>

namespace Optima {

/// A type used to describe a saddle point problem.
struct SaddlePointProblem
{
	/// The left-hand side coefficient matrix of the saddle point problem.
	SaddlePointMatrix lhs;

	/// The right-hand side vector of the saddle point problem.
	SaddlePointVector rhs;
};

/// A type used to describe a saddle point problem.
struct SaddlePointProblemCanonical
{
	/// The left-hand side coefficient matrix of the canonical saddle point problem.
	SaddlePointMatrixCanonical lhs;

	/// The right-hand side vector of the canonical saddle point problem.
	SaddlePointVectorCanonical rhs;
};

} // namespace Optima
