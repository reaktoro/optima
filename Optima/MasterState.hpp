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

// Optima includes
#include <Optima/MasterVector.hpp>

namespace Optima {

/// Used to represent the master state of an optimization problem solution.
struct MasterState
{
    MasterVector u; ///< The master vector *u = (x, p, w)*.
    Vector s;       ///< The stabilities *s* of the primal variables.
    Matrix xc;      ///< The sensitivity derivatives of *x* with respect to *c*.
    Matrix pc;      ///< The sensitivity derivatives of *p* with respect to *c*.
    Matrix wc;      ///< The sensitivity derivatives of *w* with respect to *c*.
    Matrix sc;      ///< The sensitivity derivatives of *s* with respect to *c*.

    /// Construct a default MasterState object.
    MasterState();

    /// Construct a default MasterState object.
    MasterState(const MasterDims& dims);

    /// Resise this MasterState object with given dimensions.
    auto resize(const MasterDims& dims) -> void;
};

} // namespace Optima
