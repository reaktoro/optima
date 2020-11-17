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
#include <memory>

// Optima includes
#include <Optima/MasterVector.hpp>
#include <Optima/ResidualFunction.hpp>

namespace Optima {

/// Used to compute the residual errors in the optimization calculation.
class ResidualErrors
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    VectorConstRef ex; ///< The residual errors associated with the first-order optimality conditions.
    VectorConstRef ep; ///< The residual errors associated with the external constraint equations.
    VectorConstRef ew; ///< The residual errors associated with the linear and non-linear constraint equations.
    const double& errorf; ///< The maximum residual error associated with the first-order optimality conditions.
    const double& errorv; ///< The maximum residual error associated with the external constraint equations.
    const double& errorw; ///< The maximum residual error associated with the linear and non-linear constraint equations.
    const double& error;  ///< The maximum error among all others.

    /// Construct a ResidualErrors instance.
    ResidualErrors(const MasterDims& dims);

    /// Construct a copy of a ResidualErrors instance.
    ResidualErrors(const ResidualErrors& other);

    /// Destroy this ResidualErrors instance.
    virtual ~ResidualErrors();

    /// Assign a ResidualErrors instance to this.
    auto operator=(ResidualErrors other) -> ResidualErrors&;

    /// Initialize the residual errors once before update computations.
    auto initialize(const MasterProblem& problem) -> void;

    /// Update the residual errors.
    auto update(MasterVectorView u, const ResidualFunction& F) -> void;
};

} // namespace Optima
