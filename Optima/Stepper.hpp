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
#include <Optima/Matrix.hpp>

namespace Optima {

// Forward declarations
class Constraints;
class IpSaddlePointMatrix;
class IpSaddlePointVector;
class Options;

/// The problem data needed to calculate a step using Stepper.
struct StepperProblem
{
    /// The current state of the primal variables of the canonical optimization problem.
    VectorConstRef x;

    /// The current state of the Lagrange multipliers of the canonical optimization problem.
    VectorConstRef y;

    /// The current state of the complementarity variables of the lower bounds of the canonical optimization problem.
    VectorConstRef z;

    /// The current state of the complementarity variables of the upper bounds of the canonical optimization problem.
    VectorConstRef w;

    /// The lower bound values of the canonical optimization problem.
    VectorConstRef xlower;

    /// The upper bound values of the canonical optimization problem.
    VectorConstRef xupper;

    /// The right-hand side vector of the linear equality constraints of the canonical optimization problem.
    VectorConstRef b;

    /// The gradient of the objective function.
    VectorConstRef g;

    /// The Hessian of the objective function.
    MatrixConstRef H;
};

/// The class that implements the step calculation.
class Stepper
{
public:
    /// Construct a default Stepper instance.
    Stepper();

    /// Construct a Stepper instance with given constraints.
    explicit Stepper(const Constraints& constraints);

    /// Construct a copy of an Stepper instance.
    Stepper(const Stepper& other);

    /// Destroy this Stepper instance.
    virtual ~Stepper();

    /// Assign an Stepper instance to this.
    auto operator=(Stepper other) -> Stepper&;

    /// Set the options for the step calculation.
    auto setOptions(const Options& options) -> void;

    /// Decompose the interior-point saddle point matrix used to compute the step vectors.
    auto decompose(const StepperProblem& problem) -> void;

    /// Solve the interior-point saddle point matrix used to compute the step vectors.
    /// @note Method Stepper::decompose needs to be called first.
    auto solve(const StepperProblem& problem) -> void;

    /// Return the calculated Newton step vector.
    /// @note Method Stepper::solve needs to be called first.
    auto step() const -> IpSaddlePointVector;

    /// Return the calculated residual vector for the current optimum state.
    /// @note Method Stepper::solve needs to be called first.
    auto residual() const -> IpSaddlePointVector;

    /// Return the assembled interior-point saddle point matrix.
    /// @note Method Stepper::decompose needs to be called first.
    auto matrix(const StepperProblem& problem) -> IpSaddlePointMatrix;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
