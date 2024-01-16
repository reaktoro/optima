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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to describe a matrix \eq{\begin{bmatrix}A\\J\end{bmatrix}} in canonical form.
/// The canonical form of a matrix \eq{A} is represented as:
/// \eqq{C = RAQ = \begin{bmatrix}I & S\end{bmatrix},}
/// where \eq{Q} is a permutation matrix, and \eq{R} is the *echelonizer
/// matrix* of \eq{A}. The matrices \eq{A} and \eq{J} have a behavior in which
/// \eq{A} is always constant, whereas \eq{J} varies more often. This more
/// advanced canonicalization class permits a more efficient update of the
/// canonical form of \eq{[A; J]} when \eq{J} changes. This happens when an
/// optimization problem has a non-linear equality constraint, whose Jacobian
/// changes in every iteration (the matrix \eq{J}). The matrix \eq{A}, on the
/// other hand, is related to the linear equality constraint whose coefficient
/// matrix remains the same throughout the calculation.
///
/// @warning EchelonizerExtended assumes that rows in *J* are not linearly
/// dependent on rows in *A* and vice-versa. This assumption should be sensible
/// in most application cases. However, when testing, it may happen that *W =
/// [A; J]* is produced so that this assumption is not respected.
///----------------------------------------------------------------------------------------------
class EchelonizerExtended
{
public:
    /// Construct a default EchelonizerExtended instance.
    EchelonizerExtended();

    /// Construct a EchelonizerExtended instance with given constant matrix *A* in *W = [A; J]*.
    EchelonizerExtended(MatrixView A);

    /// Construct a copy of a EchelonizerExtended instance.
    EchelonizerExtended(const EchelonizerExtended& other);

    /// Destroy this EchelonizerExtended instance.
    virtual ~EchelonizerExtended();

    /// Assign a EchelonizerExtended instance to this.
    auto operator=(EchelonizerExtended other) -> EchelonizerExtended&;

    /// Return the number of variables.
    auto numVariables() const -> Index;

    /// Return the number of equations.
    auto numEquations() const -> Index;

    /// Return the number of basic variables.
    auto numBasicVariables() const -> Index;

    /// Return the number of non-basic variables.
    auto numNonBasicVariables() const -> Index;

    /// Return the matrix \eq{S} of the canonicalization.
    auto S() const -> MatrixView;

    /// Return the echelonizer matrix \eq{R}.
    auto R() const -> MatrixView;

    /// Return the permutation matrix \eq{Q} of the canonicalization.
    /// This method returns the indices (ordering) of the variables after canonicalization.
    auto Q() const -> IndicesView;

    /// Return the canonicalized matrix \eq{C = R[A; J]Q = [I\quad S]}`.
    auto C() const -> Matrix;

    /// Return the indices of the basic variables.
    auto indicesBasicVariables() const -> IndicesView;

    /// Return the indices of the non-basic variables.
    auto indicesNonBasicVariables() const -> IndicesView;

    /// Update the canonical form with given lower matrix block *J* and priority weights for the variables.
    auto updateWithPriorityWeights(MatrixView J, VectorView weights) -> void;

    /// Update the ordering of the basic and non-basic variables.
    auto updateOrdering(IndicesView Kb, IndicesView Kn) -> void;

    /// Perform a cleanup procedure to remove residual round-off errors from the canonical form.
    auto cleanResidualRoundoffErrors() -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
