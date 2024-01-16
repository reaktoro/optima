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

/// Used to represent the echelon form of a matrix *A*.
struct EchelonMatrix
{
    MatrixView R;   ///< The echelonizer matrix of *A* so that *RAQ = [I S]* with *Q = (jb, jn)*.
    MatrixView S;   ///< The matrix *S* in the echelon form of *A*.
    IndicesView jb; ///< The indices of the basic variables in the echelon form of *A*.
    IndicesView jn; ///< The indices of the non-basic variables in the echelon form of *A*.
};

/// Used to describe a matrix \eq{A} in canonical form.
/// The canonical form of a matrix \eq{A} is represented as:
/// \eqq{C = RAQ = \begin{bmatrix}I & S\end{bmatrix},}
/// where \eq{Q} is a permutation matrix, and \eq{R} is the *echelonizer matrix* of \eq{A}.
class Echelonizer
{
public:
    /// Construct a default Echelonizer instance.
    Echelonizer();

    /// Construct a Echelonizer instance with given matrix.
    Echelonizer(MatrixView A);

    /// Construct a copy of a Echelonizer instance.
    Echelonizer(const Echelonizer& other);

    /// Destroy this Echelonizer instance.
    virtual ~Echelonizer();

    /// Assign a Echelonizer instance to this.
    auto operator=(Echelonizer other) -> Echelonizer&;

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

    /// Return the canonicalized matrix \eq{C = RAQ = [I\quad S]}`.
    auto C() const -> Matrix;

    /// Return the indices of the original equations associated with canonical equations.
    auto indicesEquations() const -> IndicesView;

    /// Return the indices of the basic variables.
    auto indicesBasicVariables() const -> IndicesView;

    /// Return the indices of the non-basic variables.
    auto indicesNonBasicVariables() const -> IndicesView;

    /// Compute the canonical matrix of the given matrix.
    auto compute(MatrixView A) -> void;

    /// Update the canonical form with the swap of a basic variable by a non-basic variable.
    /// @param ibasic The index of the basic variable between 0 and \eq{n_\mathrm{b}}`.
    /// @param inonbasic The index of the non-basic variable between 0 and \eq{n_\mathrm{n}}`.
    auto updateWithSwapBasicVariable(Index ibasic, Index inonbasic) -> void;

    /// Update the canonical form with given priority weights for the variables.
    /// This method will update the canonical form by taking into account the
    /// given priority weights of the variables when selecting the basic
    /// variables. The basic and non-basic variables will be sorted in descend
    /// order with respect to their priority weights. By choosing non-positive
    /// weights for some variables, and positive for all others, the variables
    /// with non-positive weights can be prevented from becoming basic
    /// variables. However, there is the possibility of a *degenerate case* in
    /// which one or more variables with non-positive weights need to be basic
    /// variables. This happens when all variables with non-zero coefficient in
    /// a row of matrix \eq{A} have non-positive weights.
    /// @param weights The priority weights of the variables.
    auto updateWithPriorityWeights(VectorView weights) -> void;

    /// Update the ordering of the basic and non-basic variables.
    auto updateOrdering(IndicesView Kb, IndicesView Kn) -> void;

    /// Reset to the canonical matrix form computed initially.
    /// This method exists so that after several variable swapping operations
    /// have taken place, which produce accumulated round-off errors, the first
    /// canonical form can be recovered.
    auto reset() -> void;

    /// Perform a cleanup procedure to remove residual round-off errors from the canonical form.
    auto cleanResidualRoundoffErrors() -> void;

private:
    struct Impl;

    std::unique_ptr<Impl> pimpl;
};

} // namespace Optima
