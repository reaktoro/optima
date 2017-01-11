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
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// A type that describes the requirement options in the objective function evaluation.
struct ObjectiveRequirement
{
	/// The boolean flag that indicates the need for the objective value.
	bool val = true;

	/// The boolean flag that indicates the need for the objective gradient.
	bool grad = true;

	/// The boolean flag that indicates the need for the objective Hessian.
	bool hessian = true;
};

/// A type that describes the evaluation state of an objective function
struct ObjectiveState
{
	/// The boolean flag that indicates if the objective function evaluation failed.
	bool failed = false;

    /// The evaluated value of the objective function.
    double val = 0.0;

    /// The evaluated gradient of the objective function.
    Vector grad;

    /// The evaluated Hessian of the objective function.
    Hessian hessian;

	/// The requirements in the evaluation of the objective function.
    ObjectiveRequirement requires;
};

/// A type that describes the functional signature of an objective function.
/// @param x The vector of variables.
/// @param f The evaluated state of the objective function.
using ObjectiveFunction = std::function<void(const Vector&, ObjectiveState&)>;

} // namespace Optima
