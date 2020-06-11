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
class Stability
{
public:
    /// The underlying basic data in a Stability object.
    struct Data
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

    /// Construct a default Stability object.
    Stability();

    /// Construct a Stability object with given data.
    Stability(const Data& data);

    /// Update the indices of stable and unstable variables with given data.
    auto update(const Data& data) -> void;

    /// Return the number of variables.
    auto numVariables() const -> Index;

    /// Return the number of *stable variables*.
    auto numStableVariables() const -> Index;

    /// Return the number of *unstable variables*.
    auto numUnstableVariables() const -> Index;

    /// Return the number of *lower unstable variables*.
    auto numLowerUnstableVariables() const -> Index;

    /// Return the number of *upper unstable variables*.
    auto numUpperUnstableVariables() const -> Index;

    /// Return the number of *strictly lower unstable variables*.
    auto numStrictlyLowerUnstableVariables() const -> Index;

    /// Return the number of *strictly upper unstable variables*.
    auto numStrictlyUpperUnstableVariables() const -> Index;

    /// Return the number of *strictly lower and upper unstable variables*.
    auto numStrictlyUnstableVariables() const -> Index;

    /// Return the indices of the variables ordered as *stable, lower unstable, upper unstable, strictly lower unstable, strictly upper unstable*.
    auto indicesVariables() const -> IndicesConstRef;

    /// Return the indices of the *stable variables*.
    auto indicesStableVariables() const -> IndicesConstRef;

    /// Return the indices of the *unstable variables*.
    auto indicesUnstableVariables() const -> IndicesConstRef;

    /// Return the indices of the *lower unstable variables*.
    auto indicesLowerUnstableVariables() const -> IndicesConstRef;

    /// Return the indices of the *upper unstable variables*.
    auto indicesUpperUnstableVariables() const -> IndicesConstRef;

    /// Return the indices of the *strictly lower unstable variables*.
    auto indicesStrictlyLowerUnstableVariables() const -> IndicesConstRef;

    /// Return the indices of the *strictly upper unstable variables*.
    auto indicesStrictlyUpperUnstableVariables() const -> IndicesConstRef;

    /// Return the indices of the *strictly lower and upper unstable variables*.
    auto indicesStrictlyUnstableVariables() const -> IndicesConstRef;

private:
    /// The underlying basic data with indices of stable and unstable variables.
    Data data;
};

} // namespace Optima
