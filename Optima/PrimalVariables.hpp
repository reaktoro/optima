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

// // Optima includes
// #include <Optima/Index.hpp>
// #include <Optima/Matrix.hpp>

// namespace Optima {

// // Forward declarations
// class Constraints;

// /// A type that describes the primal variables in a canonical optimization problem.
// class PrimalVariables
// {
// public:
//     /// Construct a default PrimalVariables instance.
//     PrimalVariables();

//     /// Construct a PrimalVariables instance with given constraints.
//     PrimalVariables(const Constraints& constraints);

//     /// Return the primal variables of the canonical optimization problem.
//     auto canonical() const -> VectorConstRef;

//     /// Return the primal variables of the canonical optimization problem.
//     auto canonical() -> VectorRef;

//     /// Return the primal variables of the original optimization problem.
//     auto original() const -> VectorConstRef;

//     /// Return the primal variables of the original optimization problem.
//     auto original() -> VectorRef;

//     /// Return the primal variables of the canonical optimization problem with respect to linear inequality constraints.
//     auto wrtLinearInequalityConstraints() const -> VectorConstRef;

//     /// Return the primal variables of the canonical optimization problem with respect to linear inequality constraints.
//     auto wrtLinearInequalityConstraints() -> VectorRef;

//     /// Return the primal variables of the canonical optimization problem with respect to non-linear inequality constraints.
//     auto wrtNonLinearInequalityConstraints() const -> VectorConstRef;

//     /// Return the primal variables of the canonical optimization problem with respect to non-linear inequality constraints.
//     auto wrtNonLinearInequalityConstraints() -> VectorRef;

// private:
//     /// The number of primal variables in the optimization problem
//     Index nx = 0;

//     /// The number of slack primal variables with respect to linear inequality constraints
//     Index nxli = 0;

//     /// The number of slack primal variables with respect to non-linear inequality constraints
//     Index nxni = 0;

//     /// The vector containing the primal variables of the canonical optimization problem.
//     Vector data;
// };

// } // namespace Optima
