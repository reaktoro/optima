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
#include <Optima/MasterDims.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// The sensitivity derivatives of a master optimization state.
struct MasterSensitivity
{
    Matrix xc; ///< The sensitivity derivatives \eq{\partial x/\partial c} with respect to parameters \eq{c}.
    Matrix pc; ///< The sensitivity derivatives \eq{\partial p/\partial c} with respect to parameters \eq{c}.
    Matrix wc; ///< The sensitivity derivatives \eq{\partial w/\partial c} with respect to parameters \eq{c}.
    Matrix sc; ///< The sensitivity derivatives \eq{\partial s/\partial c} with respect to parameters \eq{c}.

    /// Construct a default MasterSensitivity object.
    MasterSensitivity();

    /// Construct a MasterSensitivity object with given dimensions.
    MasterSensitivity(const MasterDims& dims, Index nc);

    /// Resise this MasterSensitivity object with given dimensions.
    auto resize(const MasterDims& dims, Index nc) -> void;
};

} // namespace Optima
