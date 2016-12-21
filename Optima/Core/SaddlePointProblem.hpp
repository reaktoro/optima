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
//// Optima includes
//#include <Optima/Core/SaddlePointMatrix.hpp>
//#include <Optima/Math/Canonicalizer.hpp>
//
//namespace Optima {
//
///// A type used to describe a saddle point problem.
//class SaddlePointProblem
//{
//public:
//    /// Construct a default SaddlePointProblem instance.
//    SaddlePointProblem();
//
//    /// Canonicalize the Jacobian matrix \eq{A} of the saddle point problem.
//    auto canonicalize(const SaddlePointMatrix& lhs) -> void;
//
//    /// Update the left-hand side coefficient matrix of the saddle point problem.
//    auto lhs(const SaddlePointMatrix& lhs) -> void;
//
//    /// Update the right-hand side vector of the saddle point problem.
//    auto rhs(const SaddlePointVector& rhs) -> void;
//
//    /// Return the left-hand side coefficient matrix of the saddle point problem in canonical and scaled form.
//    auto clhs() const -> const SaddlePointMatrixCanonical&;
//
//    /// Return the right-hand side vector of the saddle point problem in canonical and scaled form.
//    auto crhs() const -> const SaddlePointVector&;
//
//    /// Return the indices of the basic variables.
//    auto ibasic() const -> const Indices&;
//
//    /// Return the indices of the non-basic variables.
//    auto inonbasic() const -> const Indices&;
//
//    /// Return the indices of the non-basic stable variables.
//    auto istable() const -> const Indices&;
//
//    /// Return the indices of the non-basic unstable variables.
//    auto iunstable() const -> const Indices&;
//
//private:
//    struct Impl;
//
//    std::unique_ptr<Impl> pimpl;
//};
//
///// A type used to describe a saddle point problem.
//struct SaddlePointProblemCanonical
//{
//	/// The left-hand side coefficient matrix of the canonical saddle point problem.
//	SaddlePointMatrixCanonical lhs;
//
//	/// The right-hand side vector of the canonical saddle point problem.
//	SaddlePointVector rhs;
//
//	/// Compute the canonical form of a given SaddlePointProblem.
//	/// @param problem The SaddlePointProblem for which the canonical form is calculated.
//	auto compute(const SaddlePointProblem& problem) -> void;
//
//	/// Update the existing canonical form with given priority weights.
//	/// @param weights The priority, as a positive weight, of each variable to become a basic variable.
//	auto update(const Vector& weights) -> void;
//};
//
//} // namespace Optima
