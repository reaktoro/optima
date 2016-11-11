// Optima is a C++ library for computational reaction modelling.
//
// Copyright (C) 2014 Allan Leal
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
#include <Optima/Math/Matrix.hpp>

namespace Optima {

//// Forward declarations
struct OptimumOptions;
struct OptimumProblem;
struct OptimumState;

/// A type that describes the options for regularizing linear constraints.
struct RegularizerOptions
{
    /// The boolean flag that indicates if echelonization should be performed.
    /// The echelonization of the linear constraints can help on robustness and
    /// accuracy by minimizing round-off errors.
    bool echelonize = true;

    /// The maximum denominator that can exist in the coefficient matrix `A`.
    /// Set this option to zero if the coefficients in `A` are not represented
    /// by rational numbers. Otherwise, set it to the maximum denominator that can
    /// represent the coefficients in rational form. This is a useful information to
    /// eliminate round-off errors when assembling the regularized coefficient matrix.
    unsigned maxdenominator = 0;
};

/// A type that used for regularization of the linear equality constraints.
class Regularizer
{
public:
    /// Construct a default Regularizer instance.
    Regularizer();

    /// Construct a copy of an Regularizer instance
    Regularizer(const Regularizer& other);

    /// Destroy this instance
    virtual ~Regularizer();

    /// Assign an Regularizer instance to this instance
    auto operator=(Regularizer other) -> Regularizer&;

    /// Set the options for regularizing linear constraints.
    auto setOptions(const RegularizerOptions& options) -> void;

    /// Regularize the linear equality constraints of the optimum problem.
    auto regularize(const OptimumState& state, const OptimumProblem& problem) -> void;

    /// Regularize the linear equality constraints with updated equality parameters `a`.
    auto regularize(const OptimumState& state, const Vector& a) -> void;

    /// Regularize the linear equality constraints with updated optimum state only.
    auto regularize(const OptimumState& state) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
