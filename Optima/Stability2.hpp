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

// Forward declarations
class MatrixViewRWQ;

/// The arguments in method @ref Stability::update.
struct StabilityUpdateArgs
{
    MatrixViewRWQ const& RWQ; ///< The echelon form RWQ = [Ibb Sbn Sbp] of matrix W.
    VectorConstRef g;         ///< The gradient of the objective function with respect to x.
    VectorConstRef x;         ///< The values of the primal variables x.
    VectorConstRef xlower;    ///< The lower bounds of the primal variables x.
    VectorConstRef xupper;    ///< The upper bounds of the primal variables x.
};

/// The stability status of the x variables.
/// This class computes the stability of each varible in \eq{x} using
/// \eqc{s = g - W_{\mathrm{x}}^{T}\lambda,}
/// where \eq{lambda} is defined as:
/// \eqc{\lambda = R^{T}g_{\mathrm{b}}.}
/// @see Stability::status
struct StabilityStatus
{
    IndicesConstRef js;    ///< The indices of the stable variables in x.
    IndicesConstRef ju;    ///< The indices of the unstable variables in x.
    IndicesConstRef jlu;   ///< The indices of the lower unstable variables in x.
    IndicesConstRef juu;   ///< The indices of the upper unstable variables in x.
    VectorConstRef s;      ///< The stability \eq{s=g-W_{\mathrm{x}}^{T}\lambda} of the x variables.
    VectorConstRef lmbda;  ///< The canonical Lagrange multipliers \eq{lambda}. Note: `lambda` is a reserved word in python.
};

/// Used to determine the stability of the primal \eq{x} variables.
class Stability2
{
private:
    Indices jsu;   ///< The indices of the x variables ordered as jsu = (js, ju) = (js, jlu, juu).
    Index ns;      ///< The number of stable stable variables in js.
    Index nlu;     ///< The number of lower unstable stable variables in jlu.
    Index nuu;     ///< The number of upper unstable stable variables in juu.
    Vector s;      ///< The stability \eq{s=g-W_{\mathrm{x}}^{T}\lambda} of the x variables.
    Vector lambda; ///< The canonical Lagrange multipliers \eq{lambda}.

public:
    /// Construct a default Stability2 object.
    Stability2(Index nx);

    /// Update the stability status of the variables in x relative to a canonical form of matrix W.
    auto update(StabilityUpdateArgs args) -> void;

    /// Return the current stability status of the x variables.
    auto status() const -> StabilityStatus;
};

} // namespace Optima
