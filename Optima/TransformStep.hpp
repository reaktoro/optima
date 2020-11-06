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
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>
#include <Optima/TransformFunction.hpp>

namespace Optima {

/// The arguments for method @ref TransformStep::initialize.
struct TransformStepInitializeArgs
{
    /// The lower bounds for variables *x*.
    const VectorConstRef xlower;

    /// The upper bounds for variables *x*.
    const VectorConstRef xupper;

    /// The custom function that performs variable transformation.
    const TransformFunction phi;
};

/// Used to update the variables in an optimization problem using Newton steps.
class TransformStep
{
private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;

public:
    /// Construct a TransformStep object.
    TransformStep(const MasterDims& dims);

    /// Construct a copy of a TransformStep object.
    TransformStep(const TransformStep& other);

    /// Destroy this TransformStep object.
    virtual ~TransformStep();

    /// Assign a TransformStep object to this.
    auto operator=(TransformStep other) -> TransformStep&;

    /// Initialize this TransformStep object when the master optimization problem changes.
    auto initialize(TransformStepInitializeArgs args) -> void;

    /// Execute the custom transformation on the just computed state of master variables.
    auto execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> bool;
};

} // namespace Optima
