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
#include <Optima/MasterDims.hpp>
#include <Optima/CanonicalMatrix.hpp>
#include <Optima/CanonicalVector.hpp>

namespace Optima {

/// Used to solve linear problems in their canonical form.
class LinearSolverNullspace
{
public:
    /// Construct a LinearSolverNullspace instance.
    LinearSolverNullspace();

    /// Construct a copy of a LinearSolverNullspace instance.
    LinearSolverNullspace(const LinearSolverNullspace& other);

    /// Destroy this LinearSolverNullspace instance.
    virtual ~LinearSolverNullspace();

    /// Assign a LinearSolverNullspace instance to this.
    auto operator=(LinearSolverNullspace other) -> LinearSolverNullspace&;

    /// Decompose the canonical matrix.
    auto decompose(CanonicalMatrix M) -> void;

    /// Solve the linear problem in its canonical form.
    /// Using this method presumes method @ref decompose has already been
    /// called. This will allow you to reuse the decomposition of the master
    /// matrix for multiple solve computations if needed.
    /// @param M The canonical matrix in the canonical linear problem.
    /// @param a The right-hand side canonical vector in the canonical linear problem.
    /// @param[out] u The solution  vector in the canonical linear problem.
    auto solve(CanonicalMatrix M, CanonicalVectorView a, CanonicalVectorRef u) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
