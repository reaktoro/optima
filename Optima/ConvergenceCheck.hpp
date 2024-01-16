// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2023 Allan Leal
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
#include <functional>

// Optima includes
#include <Optima/MasterDims.hpp>
#include <Optima/MasterVector.hpp>
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>
#include <Optima/Result.hpp>

namespace Optima {

/// Used to store arguments for a custom additional convergence check.
struct ConvergenceCheckArgs
{
    MasterDims const& dims;
    ResidualFunction const& F;
    ResidualErrors const& E;
    MasterVector const& uo;
    MasterVector const& u;
    Result const& result;
};

/// A type that describes a custom additional convergence check.
using ConvergenceCheck = std::function<bool(ConvergenceCheckArgs const&)>;

} // namespace Optima
