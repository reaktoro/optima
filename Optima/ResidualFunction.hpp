// // Optima is a C++ library for solving linear and non-linear constrained optimization problems
// //
// // Copyright (C) 2014-2018 Allan Leal
// //
// // This program is free software: you can redistribute it and/or modify
// // it under the terms of the GNU General Public License as published by
// // the Free Software Foundation, either version 3 of the License, or
// // (at your option) any later version.
// //
// // This program is distributed in the hope that it will be useful,
// // but WITHOUT ANY WARRANTY; without even the implied warranty of
// // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// // GNU General Public License for more details.
// //
// // You should have received a copy of the GNU General Public License
// // along with this program. If not, see <http://www.gnu.org/licenses/>.

// #pragma once

// // C++ includes
// #include <memory>

// // Optima includes
// #include <Optima/MasterDims.hpp>
// #include <Optima/ConstraintFunction.hpp>
// #include <Optima/Matrix.hpp>
// #include <Optima/ObjectiveFunction.hpp>

// namespace Optima {

// // Forward declarations
// class CanonicalVector;
// class MasterVector;
// class MasterMatrix;

// /// Used to represent the arguments in ResidualFunction constructor.
// struct ResidualFunctionInitArgs
// {
//     Index nx;           ///< The number of primal variables *x*.
//     Index np;           ///< The number of parameter variables *p*.
//     Index ny;           ///< The number of Lagrange multipliers *y* (i.e. number of rows in *A = [Ax Ap]*).
//     Index nz;           ///< The number of Lagrange multipliers *z* (i.e. number of equations in *h(x, p) = 0*).
//     MatrixConstRef Ax;  ///< The coefficient matrix *Ax* of the linear equality constraints.
//     MatrixConstRef Ap;  ///< The coefficient matrix *Ap* of the linear equality constraints.
//     const ObjectiveFunction& objectivefn;   ///< The objective function *f(x, p)* of the basic optimization problem.
//     const ConstraintFunction& constraintfn; ///< The nonlinear equality constraint function *h(x, p)*.
//     const ConstraintFunction& vfn;          ///< The external nonlinear constraint function *v(x, p)*.
// };

// /// Used to represent the residual function *F(u)* in the Newton step problem.
// class ResidualFunction
// {
// public:
//     /// Construct a default ResidualFunction instance.
//     ResidualFunction(Index size);

//     /// Construct a ResidualFunction instance with given lower and upper bounds.
//     ResidualFunction(VectorConstRef lower, VectorConstRef upper);

//     /// Construct a copy of a ResidualFunction instance.
//     ResidualFunction(const ResidualFunction& other);

//     /// Destroy this ResidualFunction instance.
//     virtual ~ResidualFunction();

//     /// Assign a ResidualFunction instance to this.
//     auto operator=(ResidualFunction other) -> ResidualFunction&;

//     /// Update the residual function with given state of *u = (x, p, y, z)*.
//     auto update(const MasterVector& u) const -> void;

//     /// Update the residual function with given state of *u = (x, p, y, z)* but no Jacobian updates.
//     auto updateSkipJacobian(const MasterVector& u) const -> void;

//     auto residualVector() const -> const MasterVector&;

//     auto residualVectorCanonical() const -> const CanonicalVector&;

//     auto jacobianMatrix() const -> const MasterMatrix&;

// private:
//     struct Impl;

//     std::unique_ptr<Impl> pimpl;
// };

// } // namespace Optima
