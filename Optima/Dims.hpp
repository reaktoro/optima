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
#include <Optima/Index.hpp>

namespace Optima {

/// The class used to define the dimensions needed to setup an optimization problem.
struct Dims
{
    /// The number of primal variables in \eq{x}.
    Index x = 0;

    /// The number of unknown parameter variables in \eq{p}.
    Index p = 0;

    /// The number of linear equality constraint equations in \eq{A_{\mathrm{ex}}x+A_{\mathrm{ep}}p=b_{\mathrm{e}}}.
    Index be = 0;

    /// The number of linear inequality constraint equations in \eq{A_{\mathrm{gx}}x+A_{\mathrm{gp}}p\ge b_{\mathrm{g}}}.
    Index bg = 0;

    /// The number of non-linear equality constraint equations in \eq{h_{\mathrm{e}}(x,p)=0}.
    Index he = 0;

    /// The number of non-linear inequality constraint equations in \eq{h_{\mathrm{g}}(x,p)\geq0}.
    Index hg = 0;

    /// The number of known parameter variables in \eq{c} used to compute sensitivities.
    Index c = 0;
};

} // namespace Optima
