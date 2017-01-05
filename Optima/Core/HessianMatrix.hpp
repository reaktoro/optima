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
    HessianMatrix(const VectorXd& diagonal) : diagonal(diagonal) {}

    HessianMatrix(const MatrixXd& dense) : dense(dense) {}

    /// Convert this HessianMatrix instance to a Matrix instance.
    auto convert() const -> MatrixXd;

    Optional<VectorXd> diagonal;

    Optional<MatrixXd> dense;

//    HessianMatrix(Optional<SparseMatrix> sparse);
//    Optional<SparseMatrix> sparse;
};

///// Used to represent a Hessian matrix in various forms.
//class HessianMatrix
//{
//public:
//    /// Used to represent the possible modes for a Hessian matrix.
//    enum Mode { Zero, Diagonal, Dense, EigenDecomp };
//
//    /// Construct a default HessianMatrix instance.
//    HessianMatrix();
//
//    /// Construct a HessianMatrix instance with a given diagonal matrix.
//    HessianMatrix(const VectorXd& vec);
//
//    /// Construct a HessianMatrix instance with a given dense matrix.
//    HessianMatrix(const MatrixXd& mat);
//
//    template<typename Derived>
//    HessianMatrix(const Eigen::MatrixBase<Derived>& expr)
//    {
//        if(Derived::ColsAtCompileTime == 1)
//            diagonal(expr.rows()) = expr;
//        else dense(expr.rows()) = expr;
//    }
//
//    /// Destroy this HessianMatrix instance.
//    virtual ~HessianMatrix();
//
//    /// Set the Hessian matrix to a zero square matrix.
//    /// @param dim The dimension of the Hessian matrix.
//    auto zero(Index dim) -> void;
//
//    /// Return a reference to the diagonal of the Hessian matrix with given dimension.
//    /// @param dim The dimension of the Hessian matrix.
//    auto diagonal(Index dim) -> VectorXd&;
//
//    /// Return a const reference to the diagonal entries of the diagonal Hessian matrix.
//    auto diagonal() const -> const VectorXd&;
//
//    /// Return a reference to the dense Hessian matrix with given dimension.
//    /// @param dim The dimension of the dense Hessian matrix.
//    auto dense(Index dim) -> MatrixXd&;
//
//    /// Return a const reference to the dense Hessian matrix.
//    auto dense() const -> const MatrixXd&;
//
//    /// Return the mode of the Hessian matrix.
//    auto mode() const -> Mode;
//
//    /// Return the dimension of the Hessian matrix.
//    auto dim() const -> Index;
//
//    /// Convert this HessianMatrix instance to a Matrix instance.
//    auto convert() const -> MatrixXd;
//
//private:
//    /// The dimension of the Hessian matrix.
//    Index m_dim;
//
//    /// The current mode of this Hessian matrix.
//    Mode m_mode;
//
//    /// The diagonal vector of this diagonal Hessian matrix (if diagonal mode active).
//    VectorXd m_diagonal;
//
//    /// The matrix representing a dense Hessian matrix (if dense mode active).
//    MatrixXd m_dense;
//};

} // namespace Optima
