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
#include <Optima/LinearSolverOptions.hpp>
#include <Optima/CanonicalMatrix.hpp>
#include <Optima/CanonicalVector.hpp>
#include <Optima/MasterVector.hpp>

namespace Optima {

/// Used to solve the system of linear equations involving master matrices and master vectors.
class LinearSolver
{
public:
    /// Construct a LinearSolver instance.
    LinearSolver();

    /// Construct a copy of a LinearSolver instance.
    LinearSolver(const LinearSolver& other);

    /// Destroy this LinearSolver instance.
    virtual ~LinearSolver();

    /// Assign a LinearSolver instance to this.
    auto operator=(LinearSolver other) -> LinearSolver&;

    /// Set the options for the linear solver.
    auto setOptions(const LinearSolverOptions& options) -> void;

    /// Return the current options of this linear solver.
    auto options() const -> const LinearSolverOptions&;

    /// Decompose the canonical form of a master matrix.
    auto decompose(CanonicalMatrix Mc) -> void;

    /// Solve the linear problem.
    /// Using this method presumes method @ref decompose has already been
    /// called. This will allow you to reuse the decomposition of the canonical
    /// matrix for multiple solve computations if needed.
    /// @param Mc The canonical form of the master matrix in the linear problem.
    /// @param a The right-hand side master vector in the linear problem.
    /// @param[out] u The solution master vector in the linear problem.
    auto solve(CanonicalMatrix Mc, MasterVectorView a, MasterVectorRef u) -> void;

    /// Solve the linear problem.
    /// Using this method presumes method @ref decompose has already been
    /// called. This will allow you to reuse the decomposition of the canonical
    /// matrix for multiple solve computations if needed.
    /// @param Mc The canonical form of the master matrix in the linear problem.
    /// @param ac The right-hand side vector in the linear problem already in its canonical form.
    /// @param[out] u The solution master vector in the linear problem.
    auto solve(CanonicalMatrix Mc, CanonicalVectorView ac, MasterVectorRef u) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
