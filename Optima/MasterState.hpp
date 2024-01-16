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
#include <Optima/MasterVector.hpp>
#include <Optima/Stability.hpp>

namespace Optima {

/// Used to represent the master state of an optimization problem solution.
struct MasterState
{
    MasterVector u; ///< The master vector *u = (x, p, w)*.
    Vector s;       ///< The stability of the primal variables *x*.
    Indices js;     ///< The indices of the stable variables in *x*.
    Indices ju;     ///< The indices of the unstable variables in *x*.
    Indices jlu;    ///< The indices of the lower unstable variables in *x*.
    Indices juu;    ///< The indices of the upper unstable variables in *x*.
    Indices jb;     ///< The indices of the basic variables in *x*.
    Indices jn;     ///< The indices of the non-basic variables in *x*.

    /// Construct a default MasterState object.
    MasterState();

    /// Construct a default MasterState object.
    MasterState(const MasterDims& dims);

    /// Resise this MasterState object with given dimensions.
    auto resize(const MasterDims& dims) -> void;
};

} // namespace Optima
