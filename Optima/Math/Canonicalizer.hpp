// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// Used to describe a matrix \eq{A} in canonical form.
/// The canonical form of a matrix \eq{A} is represented as:
/// \eqq{C = RAQ = \begin{bmatrix}I & S\end{bmatrix},}
/// where \eq{Q} is a permutation matrix, and \eq{R} is the *canonicalizer matrix* of \eq{A}.
class Canonicalizer
{
public:
    /// Construct a default Canonicalizer instance.
    Canonicalizer();

    /// Construct a Canonicalizer instance with given matrix.
    Canonicalizer(const Matrix& A);

    /// Return the number of rows of the canonical form.
    auto rows() const -> Index;

    /// Return the number of columns of the canonical form.
    auto cols() const -> Index;

    /// Return the `S` matrix of the canonical representation.
    auto S() const -> const Matrix&;

    /// Return the `R` canonicalizer matrix of the canonicalization.
    auto R() const -> const Matrix&;

    /// Return the inverse of the `R` matrix of the canonicalization.
    auto Rinv() const -> const Matrix&;

    /// Return the `P` permutation matrix of the canonicalization.
    auto P() const -> const PermutationMatrix&;

    /// Return the `Q` permutation matrix of the canonicalization.
    auto Q() const -> const PermutationMatrix&;

    /// Return the indices of the linearly independent rows of the original matrix.
    auto ili() const -> Indices;

    /// Return the indices of the basic components.
    auto ibasic() const -> Indices;

    /// Return the indices of the non-basic components.
    auto inonbasic() const -> Indices;

	/// Compute the canonical matrix of the given matrix.
	auto compute(const Matrix& A) -> void;

	/// Swap a basic component by a non-basic component.
	/// Let `m` and `n` denote the number of rows and columns of
	/// the canonical matrix. The index of the basic component, `ib`,
	/// must be between zero and `m`, and the index of the non-basic
	/// component, `in`, must be between zero and `n - m`.
	/// @param ib The index of the basic component.
	/// @param in The index of the non-basic component.
	auto swap(Index ib, Index in) -> void;

	/// Update the existing canonical form with given priority weights.
    /// @param weights The priority, as a positive weight, of each variable to become a basic variable.
	auto update(const Vector& weights) -> void;

	/// Return the canonicalized matrix `C = [I S]`.
	auto matrix() const -> Matrix;

private:
    /// The matrix `S` in the canonical form `C = [I S]`.
    Matrix m_S;

    /// The permutation matrix `P`.
    PermutationMatrix m_P;

    /// The permutation matrix `Q`.
    PermutationMatrix m_Q;

    /// The canonicalizer matrix `R`.
    Matrix m_R;

    /// The inverse of the canonicalizer matrix `R`.
    Matrix m_Rinv;

    /// The matrix `M` used in the swap operation.
    Vector m_M;

    /// The permutation matrix `Kb` used in the weighted update method.
    PermutationMatrix m_Kb;

    /// The permutation matrix `Kn` used in the weighted update method.
	PermutationMatrix m_Kn;
};

} // namespace Optima
