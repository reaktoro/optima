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
#include <functional>
#include <tuple>

// Optima includes
#include <Optima/Index.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/Exception.hpp>

namespace Optima {

/// Assign matrix @p other to matrix @p mat only if both have  object to another ensuring that
template<typename Derived, typename DerivedOther>
auto assignOrError(Eigen::MatrixBase<Derived>& mat, const Eigen::MatrixBase<DerivedOther>& other) -> Eigen::MatrixBase<Derived>&
{
    if(mat.size() == 0 && other.size() == 0)
        return mat; // do not assign if both are empty but possibly with different number of rows/cols, which causes error.
    errorif(mat.rows() != other.rows(),
        "Cannot assign a matrix with ", other.rows(), " rows to another with ", mat.rows(), ".");
    errorif(mat.cols() != other.cols(),
        "Cannot assign a matrix with ", other.cols(), " columns to another with ", mat.cols(), ".");
    return mat = other;
}

/// Compute the largest step length @f$\alpha@f$ such that
/// @f$\mathbf{p} + \alpha\Delta\mathbf{p}@f$ is on the
/// lower bound @f$\mathbf{x}_l=\mathbf{0}@f$.
/// @param p The point @f$\mathbf{p}@f$
/// @param dp The step @f$\Delta\mathbf{p}@f$
/// @return The largest step length (+inf in case the step does not violate the bounds)
auto largestStep(const Vector& p, const Vector& dp) -> double;

/// Compute the largest step length @f$\alpha@f$ such that
/// @f$\mathbf{p} + \alpha\Delta\mathbf{p}@f$ does not violate
/// the given lower and upper bounds.
/// @param p The point @f$\mathbf{p}@f$
/// @param dp The step @f$\Delta\mathbf{p}@f$
/// @param plower The lower bounds for @eq{\mathbf{p}}
/// @param pupper The upper bounds for @eq{\mathbf{p}}
/// @return The largest step length (+inf in case the step does not violate the bounds)
auto largestStep(const Vector& p, const Vector& dp, const Vector& plower, const Vector& pupper) -> double;

/// Perform a conservative step using @eq{p^{\prime}=p+\alpha\Delta p}.
/// In this method, a factor @eq{\alpha\in(0,1]} is used to reduce the step
/// length of @eq{\Delta p} so that no lower or upper bounds are violated. The
/// factor @eq{alpha} is determined based on the largest bound violation.
auto performConservativeStep(Vector& p, const Vector& dp, const Vector& plower, const Vector& pupper) -> double;

/// Perform an aggressive step using @eq{p^{\prime}=\max(p_{\mathrm{lower}},\min(p,p_{\mathrm{upper}}))}.
auto performAggressiveStep(Vector& p, const Vector& dp, const Vector& plower, const Vector& pupper) -> void;

/// Compute the fraction-to-the-boundary step length given by:
/// @f[\alpha_{\mathrm{max}}=\max\{\alpha\in(0,1]:\mathbf{p}+\alpha\Delta\mathbf{p}\geq(1-\tau)\mathbf{p}\}@f.]
/// @param p The point @f$\mathbf{p}@f$
/// @param dp The step @f$\Delta\mathbf{p}@f$
/// @param tau The fraction-to-the-boundary parameter @f$\tau@f$
auto fractionToTheBoundary(const Vector& p, const Vector& dp, double tau) -> double;

/// Compute the fraction-to-the-boundary step length given by:
/// @f[\alpha_{\mathrm{max}}=\max\{\alpha\in(0,1]:\mathbf{p}+\alpha\Delta\mathbf{p}\geq(1-\tau)\mathbf{p}\}@f.]
/// @param p The point @f$\mathbf{p}@f$
/// @param dp The step @f$\Delta\mathbf{p}@f$
/// @param tau The fraction-to-the-boundary parameter @f$\tau@f$
/// @param ilimiting The index of the limiting variable
auto fractionToTheBoundary(const Vector& p, const Vector& dp, double tau, Index& ilimiting) -> double;

/// Compute the fraction-to-the-boundary step length given by:
/// @f[\alpha_{\mathrm{max}}=\max\{\alpha\in(0,1]:\alpha C\Delta\mathbf{p}\geq-\tau C\mathbf{p}+\mathbf{r}\}@f.]
/// @param p The point @f$\mathbf{p}@f$
/// @param dp The step @f$\Delta\mathbf{p}@f$
/// @param C The left-hand side matrix that defines the inequality constraint @f$C\mathbf{p}\geq\mathbf{r}@f$
/// @param r The right-hand side vector that defines the inequality constraint @f$C\mathbf{p}\geq\mathbf{r}@f$
/// @param tau The fraction-to-the-boundary parameter @f$\tau@f$
auto fractionToTheBoundary(const Vector& p, const Vector& dp, const Matrix& C, const Vector& r, double tau) -> double;

/// Compute the fraction-to-the-boundary step length given by:
/// @f[\alpha_{\mathrm{max}}=\max\{\alpha\in(0,1]:\mathbf{p}+\alpha\Delta\mathbf{p}\geq(1-\tau)\mathbf{p}\}@f.]
/// @param p The point @f$\mathbf{p}@f$
/// @param dp The step @f$\Delta\mathbf{p}@f$
/// @param lower The lower bound for the step @f$\Delta\mathbf{p}@f$
/// @param tau The fraction-to-the-boundary parameter @f$\tau@f$
auto fractionToTheLowerBoundary(const Vector& p, const Vector& dp, const Vector& lower, double tau) -> double;

/// Check if a float number is less than another by a base value.
/// This method is particularly useful when comparing two float numbers
/// where round-off errors can compromise the checking.
/// The following is used for the comparison:
/// @f[a < b + 10\epsilon \mathrm{baseval}@f],
/// where @f$\epsilon@f$ is the machine double precision.
/// @param a The left-hand side value
/// @param b The right-hand side value
/// @param baseval The base value for the comparison
auto lessThan(double a, double b, double baseval) -> bool;

/// Check if a float number is greater than another by a base value.
/// This method is particularly useful when comparing two float numbers
/// where round-off errors can compromise the checking.
/// The following is used for the comparison:
/// @f[a > b - 10\epsilon \mathrm{baseval}@f],
/// where @f$\epsilon@f$ is the machine double precision.
/// @param a The left-hand side value
/// @param b The right-hand side value
/// @param baseval The base value for the comparison
auto greaterThan(double a, double b, double baseval) -> bool;

/// Return the floating-point representation of positive infinity
constexpr auto infinity() -> double
{
    return std::numeric_limits<double>::infinity();
}

/// Return the machine epsilon
constexpr auto epsilon() -> double
{
    return std::numeric_limits<double>::epsilon();
}

/// Return an inverse Hessian function based on the BFGS Hessian approximation
auto bfgs() -> std::function<Matrix(const Vector&, const Vector&)>;

/// Calculate the minimum of a single variable function using the Golden Section Search algorithm.
auto minimizeGoldenSectionSearch(const std::function<double(double)>& f, double a, double b, double tol = 1e-5) -> double;

/// Calculate the minimum of a single variable function using the Brent algorithm.
auto minimizeBrent(const std::function<double(double)>& f, double min, double max, double tolerance = 1e-5, unsigned maxiters = 100) -> double;

/// Calculate the inverse of `A + D` where `inv(A)` is already known and `D` is a diagonal matrix.
/// @param invA[in,out] The inverse of the matrix `A` and the final inverse of `A + D`
/// @param D The diagonal matrix `D`
auto inverseShermanMorrison(const Matrix& invA, const Vector& D) -> Matrix;

/// Calculates the rational number that approximates a given real number.
/// The algorithm is based on Farey sequence as shown
/// [here](http://www.johndcook.com/blog/2010/10/20/best-rational-approximation/).
/// @param x The real number.
/// @param maxden The maximum denominator.
/// @return A tuple containing the numerator and denominator.
auto rational(double x, unsigned maxden) -> std::tuple<long, long>;

/// Rationalize the entries in a matrix/vector.
/// This method will reset the values in a matrix by the closest rational
/// number, in case the original matrix was made of rational numbers and its
/// transformation was spoiled with round-off errors.
/// @param data The pointer to the beginning of a matrix/vector data.
/// @param size The size of the matrix/vector.
/// @param maxden The maximum denominator.
auto rationalize(double* data, unsigned size, unsigned maxden) -> void;

/// Replace residual round-off errors by zeros in a matrix/vector.
/// This method replaces all entries in a matrix that are below a given
/// threshold by zeroes. These entries are considered residual round-off
/// errors, as a result of linear algebra operations on them. If a zero
/// threshold is given, a threshold is calculated as the product of epsilon
/// value and the maximum absolute value in the matrix.
auto cleanResidualRoundoffErrors(MatrixRef mat) -> void;

/// Multiply a matrix and a vector and clean residual round-off errors.
auto multiplyMatrixVectorWithoutResidualRoundOffError(MatrixView A, VectorView x) -> Vector;

/// Used to describe the structure of a matrix.
enum class MatrixStructure
{
    Zero,     ///< A matrix with zero entries only, represented by a matrix with no rows and columns.
    Dense,    ///< A matrix with no regular zero pattern, represented by a matrix with one or more rows and columns.
    Diagonal, ///< A matrix with non-zero entries only on the diagonal, represented by a matrix with a single column.
};

/// Return the structure type of the given matrix.
auto matrixStructure(MatrixView mat) -> MatrixStructure;

/// Return `true` if given matrix is a zero matrix, represented by an empty matrix.
auto isZeroMatrix(MatrixView mat) -> bool;

/// Return `true` if given matrix is a diagonal matrix, represented by a matrix with single column.
auto isDiagonalMatrix(MatrixView mat) -> bool;

/// Return `true` if given matrix is a dense matrix, represented by a matrix with more than one rows and columns.
auto isDenseMatrix(MatrixView mat) -> bool;

/// Assign a matrix with another that may be square or a single column representing a diagonal matrix.
auto operator<<=(MatrixRef mat, MatrixView other) -> MatrixRef;

/// Resize a matrix if its current dimension is inferior to a given one.
/// If both given number of rows and columns are less than the current values,
/// then no resizing is performed.
auto ensureMinimumDimension(Matrix& mat, Index rows, Index cols) -> void;

} // namespace Optima
