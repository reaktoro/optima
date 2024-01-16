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
#include <Optima/Dims.hpp>
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// The sensitivity derivatives of the optimum state.
class Sensitivity
{
public:
    Dims const dims;   ///< The dimensions of the variables and constraints in the optimization problem.
    FixedMatrix xc;    ///< The sensitivity derivatives \eq{\partial x/\partial c} with respect to parameters \eq{c}.
    FixedMatrix pc;    ///< The sensitivity derivatives \eq{\partial p/\partial c} with respect to parameters \eq{c}.
    FixedMatrix yec;   ///< The sensitivity derivatives \eq{\partial y_{\mathrm{e}/\partial c} with respect to parameters \eq{c}.
    FixedMatrix ygc;   ///< The sensitivity derivatives \eq{\partial y_{\mathrm{g}/\partial c} with respect to parameters \eq{c}.
    FixedMatrix zec;   ///< The sensitivity derivatives \eq{\partial z_{\mathrm{e}/\partial c} with respect to parameters \eq{c}.
    FixedMatrix zgc;   ///< The sensitivity derivatives \eq{\partial z_{\mathrm{g}/\partial c} with respect to parameters \eq{c}.
    FixedMatrix sc;    ///< The sensitivity derivatives \eq{\partial s/\partial c} with respect to parameters \eq{c}.
    FixedMatrix xbgc;  ///< The sensitivity derivatives \eq{\partial x_{b_{\mathrm{g}}/\partial c} with respect to parameters \eq{c}.
    FixedMatrix xhgc;  ///< The sensitivity derivatives \eq{\partial x_{h_{\mathrm{g}}/\partial c} with respect to parameters \eq{c}.

    /// Construct a default Sensitivity object.
    Sensitivity();

    /// Construct a Sensitivity object with given dimensions.
    Sensitivity(const Dims& dims);

    /// Assign a Sensitivity instance to this.
    auto operator=(const Sensitivity& other) -> Sensitivity&;

    /// Resise this Sensitivity object with given dimensions.
    auto resize(const Dims& dims) -> void;
};

} // namespace Optima
