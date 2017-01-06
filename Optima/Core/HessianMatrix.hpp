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
// on with this program. If not, see <http://www.gnu.org/licenses/>.

#pragma once

// Optima includes
#include <Optima/Common/Index.hpp>
#include <Optima/Common/Optional.hpp>
#include <Optima/Math/Matrix.hpp>

namespace Optima {

/// Used to represent a Hessian matrix in various forms.
class HessianMatrix
{
public:
    /// Used to represent the different modes of a Hessian matrix.
    enum Mode { Dense, Diagonal, Sparse };

    /// Construct a HessianMatrix instance with diagonal form.
    /// @param vec The vector representing the diagonal entries of the Hessian matrix.
    HessianMatrix(const VectorXd& vec) : m_diagonal(vec) {}

    /// Construct a HessianMatrix instance with dense form.
    /// @param mat The matrix representing the entries of the Hessian matrix.
    HessianMatrix(const MatrixXd& mat) : m_dense(mat) {}

    /// Return the dimension of the HessianMatrix instance.
    auto dim() const -> Index { return isdiagonal() ? diagonal().rows() : dense().rows(); }

    /// Return `true` if the HessianMatrix instance is in diagonal form.
    auto isdiagonal() const -> bool { return m_diagonal; }

    /// Return `true` if the HessianMatrix instance is in dense form.
    auto isdense() const -> bool { return m_dense; }

    /// Return `true` if the HessianMatrix instance is in diagonal form.
    auto diagonal() const -> const VectorXd& { return m_diagonal.value(); }

    /// Return `true` if the HessianMatrix instance is in dense form.
    auto dense() const -> const MatrixXd& { return m_dense.value(); }

    /// Return the mode of the HessianMatrix instance.
    auto mode() const -> Mode { return isdiagonal() ? Diagonal : Dense; }

    /// Convert this HessianMatrix instance into a Matrix instance.
    auto matrix() const -> MatrixXd { return isdiagonal() ? MatrixXd(diag(diagonal())) : dense(); }

private:
    /// The Hessian matrix as a diagonal matrix.
    Optional<VectorXd> m_diagonal;

    /// The Hessian matrix as a dense matrix.
    Optional<MatrixXd> m_dense;
};

/// Assign a HessianMatrix instance into a Matrix instance.
auto operator<<(MatrixRef mat, const HessianMatrix& hessian) -> MatrixRef;

} // namespace Optima
