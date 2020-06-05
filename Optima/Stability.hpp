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

/// The state of the optimization variables.
struct Stability
{
    /// The ordering of the primal variables *x* as *stable*, *lower unstable*, *upper unstable*, *strictly lower unstable*, *strictly upper unstable*.
    Indices iordering;

    /// The number of *stable variables* in *x*.
    Index ns = 0;

    /// The number of *lower unstable variables* in *x*.
    Index nlu = 0;

    /// The number of *upper unstable variables* in *x*.
    Index nuu = 0;

    /// The number of *strictly lower unstable variables* in *x*.
    Index nslu = 0;

    /// The number of *strictly upper unstable variables* in *x*.
    Index nsuu = 0;
};

} // namespace Optima
