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

// C++ includes
#include <memory>

// Optima includes
#include <Optima/ConvergenceOptions.hpp>
#include <Optima/ResidualErrors.hpp>

namespace Optima {

/// Used to perform convergence analysis during an optimization calculation.
class Convergence
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a Convergence object.
    Convergence();

    /// Construct a copy of a Convergence object.
    Convergence(const Convergence& other);

    /// Destroy this Convergence object.
    virtual ~Convergence();

    /// Assign a Convergence object to this.
    auto operator=(Convergence other) -> Convergence&;

    /// Set the options of this convergence checker.
    auto setOptions(const ConvergenceOptions& options) -> void;

    /// Initialize this convergence checker once at the start of the optimization calculation.
    auto initialize(const MasterProblem& problem) -> void;

    /// Update the convergence analysis with new accepted error status.
    auto update(const ResidualErrors& E) -> void;

    /// Return `true` if the optimization calculation has converged.
    auto converged(ConvergenceCheckArgs const& args) const -> bool;

    /// Return the current convergence rate.
    auto rate() const -> double;
};

} // namespace Optima
