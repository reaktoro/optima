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

namespace Optima {

/// Used to organize the options for convergence analysis.
struct ConvergenceOptions
{
    /// The tolerance for the optimality error.
    double tolerance = 1.0e-8;

    /// The option that controls whether convergence can only be resolved if at
    /// least one iteration was taken. The safest option is to ensure that at
    /// least one Newton step is applied, even if the equation residuals at the
    /// beginning of the calculation are below the stipulated tolerances.
    /// Otherwise, the parameters of an existing problem can be slightly changed
    /// and the corresponding solution is identical to the original problem,
    /// while a a slightly different (though very similar) new solution was
    /// expected. For example, in chemical equilibrium calculations, consider a
    /// mineral already in equilibrium with respect to an aqueous phase, and a
    /// new problem in which a small amount of the same mineral is added. When
    /// solving this new problem, the solution should reflect this small
    /// addition of the mineral. Nonetheless, the stipulated tolerance for
    /// verification of convergence would not recognize that the problem is
    /// slightly different, yielding a solution that is identical to the
    /// solution of the original problem. Mass conservation would thus not be
    /// accurate to the level of machine precision. To avoid that this more
    /// recently added control breaks existing codes, the default is false.
    double requires_at_least_one_iteration = false;
};

} // namespace Optima
