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
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to represent the canonical solution vector *us = (xs, p, wbs)* in the optimization problem.
struct CanonicalVectorRef
{
    VectorRef xs;  ///< The constant view to sub-vector *xs*.
    VectorRef p;   ///< The constant view to sub-vector *p*.
    VectorRef wbs; ///< The constant view to sub-vector *wbs*.
};

/// Used to represent the canonical solution vector *us = (xs, p, wbs)* in the optimization problem.
struct CanonicalVectorConstRef
{
    VectorConstRef xs;  ///< The constant view to sub-vector *xs*.
    VectorConstRef p;   ///< The constant view to sub-vector *p*.
    VectorConstRef wbs; ///< The constant view to sub-vector *wbs*.
};

} // namespace Optima
