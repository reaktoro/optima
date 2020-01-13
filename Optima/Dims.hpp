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
#include <Optima/Index.hpp>

namespace Optima {

/// The dimensions of variables and constraints in an optimization problem.
struct Dims
{
    /// The number of variables (equivalent to the dimension of vector \eq{x}).
    Index x = 0;

    /// The number of linear equality constraint equations (equivalent to the dimension of vector \eq{b_\mathrm{e}}).
    Index be = 0;

    /// The number of linear inequality constraint equations (equivalent to the dimension of vector \eq{b_\mathrm{i}}).
    Index bi = 0;

    /// The number of non-linear equality constraint equations (equivalent to the dimension of vector \eq{h_\mathrm{e}}).
    Index he = 0;

    /// The number of non-linear inequality constraint equations (equivalent to the dimension of vector \eq{h_\mathrm{i}}).
    Index hi = 0;

    /// The number of variables with lower bounds (equivalent to the dimension of vector of lower bounds \eq{x_\mathrm{l}}).
    Index xlower = 0;

    /// The number of variables with upper bounds (equivalent to the dimension of vector of upper bounds \eq{x_\mathrm{u}}).
    Index xupper = 0;

    /// The number of variables with fixed values (equivalent to the dimension of vector \eq{x_\mathrm{f}}).
    Index xfixed = 0;
};

} // namespace Optima
