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

// #include "Constraints.hpp"

// // Optima includes
// #include <Optima/Exception.hpp>
// #include <Optima/IndexUtils.hpp>

// namespace Optima {

// namespace internal {

// const auto defaultConstraintFunction = [](VectorConstRef x, ConstraintResult& res)
// {
// };

// } // namespace internal

// Constraints2::Constraints2()
// : Constraints2(Dims{})
// {}

// Constraints2::Constraints2(const Dims& dims)
// : Ae(dims.be, dims.x),
//   Ai(dims.bi, dims.x),
//   he(internal::defaultConstraintFunction),
//   hi(internal::defaultConstraintFunction),
//   ilower(indices(dims.xlower)),
//   iupper(indices(dims.xupper)),
//   ifixed(indices(dims.xfixed))
// {
// }


// Constraints::Constraints()
// : Constraints(0)
// {}

// Constraints::Constraints(Index n)
// : n(n), nlower(0), nupper(0), nfixed(0),
//   lowerpartition(indices(n)),
//   upperpartition(indices(n)),
//   fixedpartition(indices(n))
// {}

// auto Constraints::setEqualityConstraintMatrix(MatrixConstRef Ae_) -> void
// {
//     Assert(Ae_.cols() == n, "Could not set the equality constraint matrix.", "Mismatch number of columns and number of variables.");
//     Assert(Ae_.rows() <= n, "Could not set the equality constraint matrix.", "More linear equality constraints than number of variables.");
//     Ae = Ae_;
// }

// auto Constraints::setEqualityConstraintFunction(const ConstraintFunction& he_, Index m_he_) -> void
// {
//     Assert(m_he_ <= n, "Could not set the equality constraint function.", "More non-linear equality constraints than number of variables.");
//     he = he_;
//     m_he = m_he_;
// }

// auto Constraints::setInequalityConstraintMatrix(MatrixConstRef Ai_) -> void
// {
//     Assert(Ai_.cols() == n, "Could not set the inequality constraint matrix.", "Mismatch number of columns and number of variables.");
//     Assert(Ai_.rows() <= n, "Could not set the inequality constraint matrix.", "More linear inequality constraints than number of variables.");
//     Ai = Ai_;
// }

// auto Constraints::setInequalityConstraintFunction(const ConstraintFunction& hi_, Index m_hi_) -> void
// {
//     Assert(m_hi_ <= n, "Could not set the inequality constraint function.", "More non-linear inequality constraints than number of variables.");
//     hi = hi_;
//     m_hi = m_hi_;
// }

// auto Constraints::setVariablesWithLowerBounds(IndicesConstRef inds) -> void
// {
//     nlower = inds.size();
//     moveIntersectionRightStable(lowerpartition, inds);
// }

// auto Constraints::allVariablesHaveLowerBounds() -> void
// {
//     nlower = n;
//     lowerpartition = indices(n);
// }

// auto Constraints::setVariablesWithUpperBounds(IndicesConstRef inds) -> void
// {
//     nupper = inds.size();
//     moveIntersectionRightStable(upperpartition, inds);
// }

// auto Constraints::allVariablesHaveUpperBounds() -> void
// {
//     nupper = n;
//     upperpartition = indices(n);
// }

// auto Constraints::setVariablesWithFixedValues(IndicesConstRef inds) -> void
// {
//     nfixed = inds.size();
//     moveIntersectionRightStable(fixedpartition, inds);
// }

// auto Constraints::numVariables() const -> Index
// {
//     return n;
// }

// auto Constraints::numLinearEqualityConstraints() const -> Index
// {
//     return Ae.rows();
// }

// auto Constraints::numLinearInequalityConstraints() const -> Index
// {
//     return Ai.rows();
// }

// auto Constraints::numNonLinearEqualityConstraints() const -> Index
// {
//     return m_he;
// }

// auto Constraints::numNonLinearInequalityConstraints() const -> Index
// {
//     return m_hi;
// }

// auto Constraints::equalityConstraintMatrix() const -> MatrixConstRef
// {
//     return Ae;
// }

// auto Constraints::equalityConstraintFunction() const -> const ConstraintFunction&
// {
//     return he;
// }

// auto Constraints::inequalityConstraintMatrix() const -> MatrixConstRef
// {
//     return Ai;
// }

// auto Constraints::inequalityConstraintFunction() const -> const ConstraintFunction&
// {
//     return hi;
// }

// auto Constraints::variablesWithLowerBounds() const -> IndicesConstRef
// {
//     return lowerpartition.tail(nlower);
// }

// auto Constraints::variablesWithUpperBounds() const -> IndicesConstRef
// {
//     return upperpartition.tail(nupper);
// }

// auto Constraints::variablesWithFixedValues() const -> IndicesConstRef
// {
//     return fixedpartition.tail(nfixed);
// }

// auto Constraints::variablesWithoutLowerBounds() const -> IndicesConstRef
// {
//     return lowerpartition.head(n - nlower);
// }

// auto Constraints::variablesWithoutUpperBounds() const -> IndicesConstRef
// {
//     return upperpartition.head(n - nupper);
// }

// auto Constraints::variablesWithoutFixedValues() const -> IndicesConstRef
// {
//     return fixedpartition.head(n - nfixed);
// }

// auto Constraints::orderingLowerBounds() const -> IndicesConstRef
// {
//     return lowerpartition;
// }

// auto Constraints::orderingUpperBounds() const -> IndicesConstRef
// {
//     return upperpartition;
// }

// auto Constraints::orderingFixedValues() const -> IndicesConstRef
// {
//     return fixedpartition;
// }

// } // namespace Optima
