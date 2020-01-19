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

// C++ includes
#include <any>

// Optima includes
#include <Optima/Matrix.hpp>
#include <Optima/ObjectiveFunction.hpp>

namespace Optima {

/// The parameters of an optimization problem that change with more frequency.
class Params
{
public:
    /// The right-hand side vector of the linear equality constraints \eq{A_{\mathrm{e}}x = b_{\mathrm{e}}}.
    Vector be;

    /// The right-hand side vector of the linear inequality constraints \eq{A_{\mathrm{i}}x = b_{\mathrm{i}}}.
    Vector bi;

    /// The lower bounds of the variables in \eq{x} that have lower bounds.
    Vector xlower;

    /// The upper bounds of the variables \eq{x} that have upper bounds.
    Vector xupper;

    /// The values of the variables in \eq{x} that are fixed.
    Vector xfixed;

    /// The extra parameters in the problem.
    std::any extra;
};

} // namespace Optima