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
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>

namespace Optima {

/// Used to describe a matrix \eq{\begin{bmatrix}A\\J\end{bmatrix}} in canonical form.
/// The canonical form of a matrix \eq{A} is represented as:
/// \eqq{C = RAQ = \begin{bmatrix}I & S\end{bmatrix},}
/// where \eq{Q} is a permutation matrix, and \eq{R} is the *canonicalizer matrix* of \eq{A}.
/// The matrices \eq{A} and \eq{J} have a behavior in which \eq{A}
/// is always constant, whereas \eq{J} varies more often. This more advanced canonicalization
/// class permits a more efficient update of the canonical form of \eq{[A; J]} when \eq{J} changes.
/// This happens when an optimization problem has a non-linear equality constraint, whose Jacobian
/// changes in every iteration (the matrix \eq{J}). The matrix \eq{A}, on the other hand, is related
/// to the linear equality constraint whose coefficient matrix remains the same throughout the calculation.
class CanonicalizerAdvanced
{
public:
    /// Construct a default CanonicalizerAdvanced instance.
    CanonicalizerAdvanced();

    /// Construct a CanonicalizerAdvanced instance with given matrix \eq{A}.
    CanonicalizerAdvanced(MatrixConstRef A);

    /// Construct a copy of a CanonicalizerAdvanced instance.
    CanonicalizerAdvanced(const CanonicalizerAdvanced& other);

    /// Destroy this CanonicalizerAdvanced instance.
    virtual ~CanonicalizerAdvanced();

    /// Assign a CanonicalizerAdvanced instance to this.
    auto operator=(CanonicalizerAdvanced other) -> CanonicalizerAdvanced&;

    /// Return the number of variables.
    auto numVariables() const -> Index;

    /// Return the number of equations.
    auto numEquations() const -> Index;

    /// Return the number of basic variables.
    auto numBasicVariables() const -> Index;

    /// Return the number of non-basic variables.
    auto numNonBasicVariables() const -> Index;

    /// Return the matrix \eq{S} of the canonicalization.
    auto S() const -> MatrixConstRef;

    /// Return the canonicalizer matrix \eq{R}.
    auto R() const -> MatrixConstRef;

    /// Return the permutation matrix \eq{Q} of the canonicalization.
    /// This method returns the indices (ordering) of the variables after canonicalization.
    auto Q() const -> IndicesConstRef;

    /// Return the canonicalized matrix \eq{C = R[A; J]Q = [I\quad S]}`.
    auto C() const -> Matrix;

    /// Return the indices of the linearly independent rows of the original matrix.
    auto indicesLinearlyIndependentEquations() const -> IndicesConstRef;

    /// Return the indices of the basic variables.
    auto indicesBasicVariables() const -> IndicesConstRef;

    /// Return the indices of the non-basic variables.
    auto indicesNonBasicVariables() const -> IndicesConstRef;

    /// Compute the canonical matrix with given matrix \eq{A}.
    auto compute(MatrixConstRef A) -> void;

    /// Update the canonical form with given matrix \eq{J} and priority weights for the variables.
    auto update(MatrixConstRef J, VectorConstRef weights) -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
