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
#include <Optima/Constraints.hpp>
#include <Optima/Objective.hpp>

namespace Optima {

/// The definition of the optimization problem.
class Problem
{
public:
   /// Construct a default Problem instance.
   Problem();

   /// Construct a Problem instance with given objective and constraints.
   /// @param objective The objective function of the optimization problem.
   /// @param constraints The constraints of the optimization problem.
   Problem(const ObjectiveFunction& objective, const Constraints& constraints);

   // /// Return right-hand side vector of the equality constraint \eq{Ax = b}.
   // auto b() -> VectorRef { return m_b; }

   // /// Return right-hand side vector of the equality constraint \eq{Ax = b}.
   // auto b() const -> VectorConstRef { return m_b; }

   // /// Set a common lower bound value for the variables \eq{x}.
   // auto xlower(double val) -> void;

   // /// Set the lower bounds of the variables \eq{x}.
   // /// @param values The values of the lower bounds.
   // auto xlower(VectorConstRef values) -> void;

   // /// Set the lower bounds of selected variables in \eq{x}.
   // /// @param indices The indices of the variables in \eq{x} with lower bounds.
   // /// @param values The values of the lower bounds.
   // auto xlower(IndicesConstRef indices, VectorConstRef values) -> void;

   // /// Return the lower bounds of the variables \eq{x}.
   // auto xlower() const -> VectorConstRef;

   // /// Set a common upper bound value for the variables \eq{x}.
   // auto xupper(double val) -> void;

   // /// Set the upper bounds of the variables \eq{x}.
   // /// @param values The values of the upper bounds.
   // auto xupper(VectorConstRef values) -> void;

   // /// Set the upper bounds of selected variables in \eq{x}.
   // /// @param indices The indices of the variables in \eq{x} with upper bounds.
   // /// @param values The values of the upper bounds.
   // auto xupper(IndicesConstRef indices, VectorConstRef values) -> void;

   // /// Return the upper bounds of the variables \eq{x}.
   // auto xupper() const -> VectorConstRef;

   // /// Set a common value for the fixed variables in \eq{x}.
   // auto xfixed(double val) -> void;

   // /// Set the values of the fixed variables in \eq{x}.
   // /// @param values The values of the fixed variables.
   // auto xfixed(VectorConstRef values) -> void;

   // /// Set the fixed values of selected variables in \eq{x}.
   // /// @param indices The indices of the fixed variables in \eq{x}.
   // /// @param values The values of the fixed variables.
   // auto xfixed(IndicesConstRef indices, VectorConstRef values) -> void;

   // /// Return the values of the fixed variables in \eq{x}.
   // auto xfixed() const -> VectorConstRef;

private:
   /// The objective function.
   ObjectiveFunction m_objective;

   /// The objective function.
   Constraints m_constraints;

   /// The right-hand side vector of the linear equality constraint \eq{A_{e}x = b_{e}}.
   Vector m_be;

   /// The lower bounds of the variables \eq{x}.
   Vector m_xlower;

   /// The upper bounds of the variables \eq{x}.
   Vector m_xupper;

   /// The values of the variables in \eq{x} that are fixed.
   Vector m_xfixed;
};

} // namespace Optima
