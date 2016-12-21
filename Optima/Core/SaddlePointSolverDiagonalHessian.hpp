//// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
////
//// Copyright (C) 2014-2017 Allan Leal
////
//// This program is free software: you can redistribute it and/or modify
//// it under the terms of the GNU General Public License as published by
//// the Free Software Foundation, either version 3 of the License, or
//// (at your option) any later version.
////
//// This program is distributed in the hope that it will be useful,
//// but WITHOUT ANY WARRANTY; without even the implied warranty of
//// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
//// GNU General Public License for more details.
////
//// You should have received a copy of the GNU General Public License
//// along with this program. If not, see <http://www.gnu.org/licenses/>.
//
//#pragma once
//
//// C++ includes
//#include <memory>
//
//// Optima includes
//#include <Optima/Core/SaddlePointMatrix.hpp>
//#include <Optima/Core/SaddlePointProblem.hpp>
//
//namespace Optima {
//
//class SaddlePointSolverDiagonalHessian
//{
//public:
//    /// Construct a default SaddlePointSolverDiagonalHessian instance.
//    SaddlePointSolverDiagonalHessian();
//
//    /// Construct a copy of a SaddlePointSolverDiagonalHessian instance.
//    SaddlePointSolverDiagonalHessian(const SaddlePointSolverDiagonalHessian& other);
//
//    /// Destroy this SaddlePointSolverDiagonalHessian instance.
//    virtual ~SaddlePointSolverDiagonalHessian();
//
//    /// Assign a SaddlePointSolverDiagonalHessian instance to this.
//    auto operator=(SaddlePointSolverDiagonalHessian other) -> SaddlePointSolverDiagonalHessian&;
//
//    /// Decompose the coefficient matrix of the canonical saddle point problem with diagonal Hessian matrix.
//    /// @param clhs The coefficient matrix of the canonical saddle point problem.
//    auto decompose(const SaddlePointMatrixCanonical& clhs) -> void;
//
//    /// Solve the canonical saddle point problem with diagonal Hessian matrix.
//    /// @note This method expects that a call to method @ref decompose has already been performed.
//    /// @param clhs The coefficient matrix of the canonical saddle point problem.
//    /// @param crhs The right-hand side vector of the canonical saddle point problem.
//    /// @param csol The solution of the canonical saddle point problem.
//    auto solve(const SaddlePointMatrixCanonical& clhs, const SaddlePointVector& crhs, SaddlePointVector& csol) -> void;
//
//private:
//    struct Impl;
//
//    std::unique_ptr<Impl> pimpl;
//};
//
//} // namespace Optima
