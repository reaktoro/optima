// Optima is a C++ library for computational reaction modelling.
//
// Copyright (C) 2014 Allan Leal
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

#include "Regularizer.hpp"

// Optima include
#include <Optima/Common/Exception.hpp>
#include <Optima/Common/Index.hpp>
#include <Optima/Common/SetUtils.hpp>
#include <Optima/Core/OptimumOptions.hpp>
#include <Optima/Core/OptimumProblem.hpp>
#include <Optima/Core/OptimumState.hpp>
#include <Optima/Math/LU.hpp>
#include <Optima/Math/Matrix.hpp>
#include <Optima/Math/Utils.hpp>

namespace Optima {
namespace {

template<typename Matrix, typename Vector>
auto determineQuasiFeasibleConstraints(
	const Matrix& A,
	const Vector& a,
	const Vector& l,
	const Vector& u,
	Indices& iquasifeasible_lower,
	Indices& iquasifeasible_upper) -> void
{
    // Clear the output vectors
	iquasifeasible_lower.clear();
	iquasifeasible_upper.clear();

	for(Index i = 0; i < A.rows(); ++i)
	{
		if(min(A.row(i)) >= 0.0) // if all entries in A.row(i) are non-negative
		{
			if(l.rows() && A.row(i)*l >= a[i])
				iquasifeasible_lower.push_back(i);
			if(u.rows() && A.row(i)*u <= a[i])
				iquasifeasible_upper.push_back(i);
		}
		else if(max(A.row(i)) < 0.0) // if all entries in A.row(i) are negative
		{
			if(l.rows() && A.row(i)*l <= a[i])
				iquasifeasible_lower.push_back(i);
			if(u.rows() && A.row(i)*u >= a[i])
				iquasifeasible_upper.push_back(i);
		}
	}
}

template<typename Matrix>
auto determineLowerUpperFixedVariables(
	const Matrix& A,
	const Indices& iquasifeasible_lower,
	const Indices& iquasifeasible_upper,
	Indices& ifixed_lower,
	Indices& ifixed_upper) -> void
{
    ifixed_lower.clear;
    ifixed_upper.clear;
    for(auto i : iquasifeasible_lower)
        for(Index = j = 0; j < A.cols(); ++j)
            if(A(i, j) != 0.0)
                ifixed_lower.push_back(j);
}

} // namespace
struct Regularizer::Impl
{
    /// The parameters for the regularization
    RegularizerOptions options;

    OptimumProblem problem;

    /// The columns of A corresponding to free variables
    Matrix Afree;

    /// The columns of A corresponding to fixed variables
    Matrix Afixed;

    //=============================================================================================
    // Data related to trivial and linearly dependent constraints.
    // Note that these members do not need to be recomputed if matrix A does not change.
    //=============================================================================================
    /// The indices of the variables fixed at their lower bounds because of linear equality equations quasi-feasible from below.
    Indices ifixed_lower_bounds;

    /// The indices of the variables fixed at their upper bounds because of linear equality equations quasi-feasible from above.
    Indices ifixed_upper_bounds;

    /// The indices of the linear equality constraints quasi-feasible from below.
    Indices iconstraints_quasifeasible_lower;

    /// The indices of the linear equality constraints quasi-feasible from above.
    Indices iconstraints_quasifeasible_above;


    /// The indices of the linearly independent constraints
    Indices ili_constraints;

    /// The permutation matrix used to order linearly independent rows.
    PermutationMatrix P_li;

    /// The flag that indicates if all non-trivial constraints are linearly independent
    bool all_li;

    /// The number of non-trivial, linearly independent rows.
    Index m_li;

    /// The coefficient matrix `A` without trivial constraints and linearly dependent rows.
    /// If new `A` in `update` equals `A_last`, then there is no need to update `A_star`.
    Matrix A_star;

    /// The right-hand side vector `b` without trivial constraints and linearly dependent rows.
    /// If new `A` in `update` equals `A_last`, then there is no need to update `b_star`.
    Matrix b_star;

    //=============================================================================================
    // Data related to echelonization of the constraints (helps with round-off errors).
    // These members are calculated from `A_star` and `b_star`.
    //=============================================================================================
    /// The weights computed for the regularization.
    Vector W;

    /// The regularizer matrix that is applied to the coefficient matrix `A_star` as `reg(A) = R*A_star`.
    /// If the new set of basic variables are the same as last, then there is no need to update `R`.
    Matrix R;

    /// The inverse of the regularizer matrix R.
    /// If the new set of basic variables are the same as last, then there is no need to update `invR`.
    Matrix invR;

    /// The permutation matrix from the echelonization.
    PermutationMatrix P_echelon;

    /// The coefficient matrix computed as `A_echelon = R * A_star`.
    /// If the new set of basic variables are the same as last, then there is no need to update `A_echelon`.
    Matrix A_echelon;

    /// The right-hand side vector `b_echelon` computed as `b_echelon = R * b_star`.
    /// This member must always be updated because `b` changes frequently.
    Matrix b_echelon;

    ///// The indices of basic/independent variables that compose the others.
    Indices ibasic_variables;

    /// The indices of basic/independent variables that compose the others in the last call.
    Indices ibasic_variables_last;

    /// The full-pivoting LU decomposition of the coefficient matrices `A*` and `A(echelon)`.
    LU lu_star, lu_echelon;

    /// Determine the trivial constraints and trivial variables.
    /// Trivial constraints are all those which fix the values of
    /// some variables (trivial variables) to the bounds.
    /// This method should be called after `determineIfSameConstraints`.
    auto determineTrivialConstraints(const OptimumProblem& problem) -> void;

    /// Determine the values of the trivial variables.
    /// This method should be called after `determineTrivialConstraints`.
    auto determineTrivialVariables(const OptimumProblem& problem) -> void;

    /// Determine the linearly dependent constraints.
    /// This method should be called only after `determineTrivialConstraints`.
    auto determineLinearlyDependentConstraints(const OptimumProblem& problem) -> void;

    /// Assemble the constraints in cannonical form to help in the prevention of round-off errors.
    /// This method should be called only after `determineLinearlyDependentConstraints`.
    auto assembleEchelonConstraints(const OptimumState& state) -> void;

    /// Remove all trivial constraints from the optimum problem.
    /// This method also adjust the members in optimum state and options that
    /// correspond to the removed trivial constraints and trivial variables.
    /// This method should be called after `assembleEchelonConstraints`
    auto removeTrivialConstraints(OptimumProblem& problem, OptimumState& state, OptimumOptions& options) -> void;

    /// Remove all linearly dependent constraints from the optimum problem.
    /// This method also adjust the members in optimum state and options that
    /// correspond to the removed linear constraints.
    /// This method should be called after `removeTrivialConstraints`
    auto removeLinearlyDependentConstraints(OptimumProblem& problem, OptimumState& state, OptimumOptions& options) -> void;

    /// Transform the original linear constraints into a echelon form as a way to minimize round-off errors.
    /// This method should be called after `fixInfeasibleConstraints`
    auto echelonizeConstraints(OptimumProblem& problem, OptimumState& state, OptimumOptions& options) -> void;

    /// Update the linear constraints of an optimum problem.
    /// This is the last method to be called during the regularization steps.
    /// It should be called after `echelonizeConstraints`.
    auto updateConstraints(OptimumProblem& problem) -> void;

    /// Fix the infeasible linear constraints by adjusting their right-hand side values.
    /// This is the last method to be called during the regularization steps.
    auto fixInfeasibleConstraints(OptimumProblem& problem) -> void;

    /// Regularize the optimum problem, state, and options before they are used in an optimization calculation.
    auto regularize(OptimumProblem& problem, OptimumState& state, OptimumOptions& options) -> void;

    /// Regularize the vectors `dg/dp` and `db/dp`, where `g = grad(f)`.
    auto regularize(Vector& dgdp, Vector& dbdp) -> void;

    /// Recover an optimum state to an state that corresponds to the original optimum problem.
    auto recover(OptimumState& state) -> void;

    /// Recover the sensitivity derivative `dxdp`.
    auto recover(Vector& dxdp) -> void;
};

auto Regularizer::Impl::determineTrivialConstraints(const OptimumProblem& problem) -> void
{
    // Auxiliary references
    const Matrix& A = problem.A;
    const Vector& a = problem.a;
    const Vector& l = problem.xlower;

    // The number of rows and cols in the original coefficient matrix
    const Index m = A.rows();
    const Index n = A.cols();

    // Clear previous states of trivial and non-trivial constraints and variables
    izeroconstraints.clear();
    ifixedlowerbounds.clear();
    inontrivial_constraints.clear();
    inontrivial_variables.clear();

    // Return true if the i-th constraint forces the variables to be fixed on the lower bounds
    auto istrivial = [&](Index irow)
    {
        return ( min(A.row(irow)) >= 0 && A.row(irow)*l >= a[irow] ) ||
               ( max(A.row(irow)) <= 0 && A.row(irow)*l <= a[irow] );
    };

    // Determine the original equality constraints that fix variables on the lower bound
    for(Index i = 0; i < m; ++i)
        if(istrivial(i))
            izeroconstraints.push_back(i);

    // Skip the rest if there are no trivial constraints
    if(izeroconstraints.size())
    {
        // Determine the original trivial variables that are fixed at their lower bounds
        for(Index i = 0; i < n; ++i)
            for(Index j = 0; j < m; ++j)
                if(A(j, i) != 0.0 && contained(j, izeroconstraints))
                    { ifixedlowerbounds.push_back(i); break; }

        // Update  the indices of the non-trivial original constraints
        inontrivial_constraints = difference(range(m), izeroconstraints);

        // Update the indices of the non-trivial original variables
        inontrivial_variables = difference(range(n), ifixedlowerbounds);

        // Initialize the matrix `A_star` by removing trivial constraints and variables
        A_star = submatrix(A, inontrivial_constraints, inontrivial_variables);
    }
    else
    {
        // Initialize the matrix `A_star` as the original matrix A
        A_star = A;
    }
}

auto Regularizer::Impl::determineTrivialVariables(const OptimumProblem& problem) -> void
{
    xtrivial = rows(problem.xlower, ifixedlowerbounds);
}

auto Regularizer::Impl::determineLinearlyDependentConstraints(const OptimumProblem& problem) -> void
{
    // The number of rows and cols in the coefficient matrix A*,
    // i.e., the original A matrix with removed trivial constraints and variables
    const Index m = A_star.rows();
    const Index n = A_star.cols();

    // Compute the LU decomposition of A*
    lu_star.compute(A_star);

    // Auxiliary references to LU components
    const auto& P = lu_star.P;
    const auto& rank = lu_star.rank;

    // Check if all constraints are linearly independent
    all_li = rank == m;

    // Skip the rest if all non-trivial constraints are linearly independent
    if(all_li)
        return;

    // Initialize the indices of the linearly independent constraints
    ili_constraints = Indices(P.indices().data(), P.indices().data() + rank);

    // Update the permutation matrix and the number of linearly independent constraints
    P_li = lu_star.P;
    m_li = lu_star.rank;

    // Permute the rows of A and remove the linearly dependent ones
    A_star = P_li * A_star;
    A_star.conservativeResize(m_li, n);
}

auto Regularizer::Impl::assembleEchelonConstraints(const OptimumState& state) -> void
{
    // Skip if echelonization should not be performed
    if(!options.echelonize)
        return;

    // Initialize the weight vector `W`
    W = log(state.x);
    W = W - min(W) + 1;

    // Remove all components in W corresponding to trivial variables
    if(izeroconstraints.size())
        W = rows(W, inontrivial_variables);

    // Compute the LU decomposition of matrix A_star with column-sorting weights.
    // Columsn corresponding to variables with higher weights are moved to the beginning of the matrix.
    // This gives preference for those variables to become linearly independent basis
    lu_echelon.compute(A_star, W);

    // Auxiliary references to LU components
    const auto& P = lu_echelon.P;
    const auto& Q = lu_echelon.Q;
    const auto& rank = lu_echelon.rank;

    // Initialize the indices of the basic variables
    ibasic_variables = Indices(Q.indices().data(), Q.indices().data() + rank);

    // Check if the new set of basic variables is diffent than the previous
    if(!equal(ibasic_variables, ibasic_variables_last))
    {
        // Update the last set of basic variables
        ibasic_variables_last = ibasic_variables;

        // The rank of the original coefficient matrix
        const auto r = lu_echelon.rank;

        // The L factor of the original coefficient matrix
        const auto L = lu_echelon.L.topLeftCorner(r, r).triangularView<Eigen::Lower>();

        // The U1 part of U = [U1 U2]
        const auto U1 = lu_echelon.U.topLeftCorner(r, r).triangularView<Eigen::Upper>();

        // Compute the regularizer matrix R = inv(U1)*inv(L)
        R = identity(r, r);
        R = L.solve(R);
        R = U1.solve(R);

        // Compute the inverse of the regularizer matrix inv(R) = L*U1
        invR = U1;
        invR = L * invR;

        // Update the permutation matrix in the echelonization
        P_echelon = P;

        // Compute the equality constraint regularization
        A_echelon = P_echelon * A_star;
        A_echelon = R * A_echelon;

        // Check if the regularizer matrix is composed of rationals.
        // If so, round-off errors can be eliminated
        if(options.maxdenominator)
        {
            cleanRationalNumbers(A_echelon, options.maxdenominator);
            cleanRationalNumbers(R, options.maxdenominator);
            cleanRationalNumbers(invR, options.maxdenominator);
        }
    }
}

auto Regularizer::Impl::removeTrivialConstraints(
    OptimumProblem& problem, OptimumState& state, OptimumOptions& options) -> void
{
    // Skip the rest if there is no trivial constraint
    if(izeroconstraints.empty())
        return;

    // The auxiliary vector used in the lambda functions below.
    // The use of this vector `x` ensures that trivial components
    // remain unchanged on the lower bounds.
    Vector x = problem.xlower;

    // Set the number of primal variables as the number of non-trivial variables
    problem.n = inontrivial_variables.size();

    // Remove trivial components from problem.b
    problem.a = rows(problem.a, inontrivial_constraints);

    // Remove trivial components from problem.c
    if(problem.c.rows())
        problem.c = rows(problem.c, inontrivial_variables);

    // Remove trivial components from problem.l
    if(problem.xlower.rows())
        problem.xlower = rows(problem.xlower, inontrivial_variables);

    // Remove trivial components from problem.u
    if(problem.xupper.rows())
        problem.xupper = rows(problem.xupper, inontrivial_variables);

    // Remove trivial components from problem.objective
    if(problem.objective)
    {
        // The objective function before it is regularized.
        ObjectiveFunction original_objective = problem.objective;

        // The evaluation of the regularized objective function
        ObjectiveState faux;

        // Update the objective function
        problem.objective = [=](const Vector& X, ObjectiveState& f) mutable
        {
            rows(x, inontrivial_variables) = X;

            original_objective(x, faux);

            f.failed = faux.failed;
            f.val = faux.val;
            f.grad = rows(faux.grad, inontrivial_variables);
            f.hessian.mode = faux.hessian.mode;

            if(faux.hessian.dense.size())
                f.hessian.dense = submatrix(faux.hessian.dense, inontrivial_variables, inontrivial_variables);
            if(faux.hessian.diagonal.size())
                f.hessian.diagonal = rows(faux.hessian.diagonal, inontrivial_variables);
            if(faux.hessian.inverse.size())
                f.hessian.inverse = submatrix(faux.hessian.inverse, inontrivial_variables, inontrivial_variables);
        };
    }

    // Remove trivial components corresponding to trivial variables and trivial constraints
    state.x = rows(state.x, inontrivial_variables);
    state.y = rows(state.y, inontrivial_constraints);
    state.z = rows(state.z, inontrivial_variables);

    // Update the names of the constraints and variables accordingly
    if(options.output.active)
    {
    	if(options.output.xnames.size())
    		options.output.xnames = extract(options.output.xnames, inontrivial_variables);
    	if(options.output.ynames.size())
    		options.output.ynames = extract(options.output.ynames, inontrivial_constraints);
    	if(options.output.znames.size())
    		options.output.znames = extract(options.output.znames, inontrivial_variables);
    }
}

auto Regularizer::Impl::removeLinearlyDependentConstraints(
    OptimumProblem& problem, OptimumState& state, OptimumOptions& options) -> void
{
    // Skip the rest if A* has all rows linearly independent
    if(all_li)
        return;

    // Remove the components in b corresponding to linearly dependent constraints
    problem.a = P_li * problem.a;
    problem.a.conservativeResize(m_li);

    // Remove the components in y corresponding to linearly dependent constraints
    state.y = P_li * state.y;
    state.y.conservativeResize(m_li);

    // Update the names of the dual components y
    if(options.output.active)
    	if(options.output.ynames.size())
    		options.output.ynames = extract(options.output.ynames, ili_constraints);
}

auto Regularizer::Impl::echelonizeConstraints(
    OptimumProblem& problem, OptimumState& state, OptimumOptions& options) -> void
{
    // Skip if echelonization should not be performed or A(echelon) was not computed
    if(!options.echelonize || !A_echelon.size())
        return;

    // Compute the corresponding echelonized right-hand side vector b
    problem.a = P_echelon * problem.a;
    problem.a = R * problem.a;

    // Update the y-Lagrange multipliers that correspond now to basic variables
    state.y = P_echelon * state.y;
    state.y = tr(invR) * state.y;

    // Update the names of the constraints to the names of basic variables
    if(options.output.active)
		if(options.output.xnames.size())
			options.output.ynames = extract(options.output.xnames, ibasic_variables);
}

auto Regularizer::Impl::updateConstraints(OptimumProblem& problem) -> void
{
    problem.A = options.echelonize && A_echelon.size() ? A_echelon : A_star;
}

auto Regularizer::Impl::fixInfeasibleConstraints(OptimumProblem& problem) -> void
{
    // Auxiliary variables
    const Index m = problem.A.rows();

    // Auxiliary references
    const auto& A = problem.A;
    const auto& l = problem.xlower;
    auto& a = problem.a;

    // Fix any right-hand side that is infeasible
    for(Index i = 0; i < m; ++i)
        if(min(A.row(i)) >= 0 && min(l) >= 0)
            a[i] = std::max(a[i], dot(A.row(i), l));
        else if(max(A.row(i)) <= 0 && max(l) >= 0)
            a[i] = std::min(a[i], dot(A.row(i), l));
}

auto Regularizer::Impl::regularize(OptimumProblem& problem, OptimumState& state, OptimumOptions& options) -> void
{
    determineTrivialConstraints(problem);
    determineTrivialVariables(problem);
    determineLinearlyDependentConstraints(problem);
    assembleEchelonConstraints(state);

    removeTrivialConstraints(problem, state, options);
    removeLinearlyDependentConstraints(problem, state, options);
    echelonizeConstraints(problem, state, options);
    updateConstraints(problem);
    fixInfeasibleConstraints(problem);
}

auto Regularizer::Impl::regularize(Vector& dgdp, Vector& dbdp) -> void
{
    // Remove derivative components corresponding to trivial constraints
    if(izeroconstraints.size())
    {
        dbdp = rows(dbdp, inontrivial_constraints);
        dgdp = rows(dgdp, inontrivial_variables);
    }

    // If there are linearly dependent constraints, remove corresponding components
    if(!all_li)
    {
        dbdp = P_li * dbdp;
        dbdp.conservativeResize(m_li);
    }

    // Perform echelonization of the right-hand side vector if needed
    if(options.echelonize && A_echelon.size())
    {
        dbdp = P_echelon * dbdp;
        dbdp = R * dbdp;
    }
}

auto Regularizer::Impl::recover(OptimumState& state) -> void
{
    // Calculate dual variables y w.r.t. original equality constraints
    state.y = lu_star.trsolve(state.f.grad - state.z);

    // Check if there was any trivial variables and update state accordingly
    if(ifixedlowerbounds.size())
    {
        // Define some auxiliary size variables
        const Index nn = inontrivial_variables.size();
        const Index nt = ifixedlowerbounds.size();
        const Index mn = inontrivial_constraints.size();
        const Index mt = izeroconstraints.size();
        const Index n = nn + nt;
        const Index m = mn + mt;

        // Set the final objective evaluation
        state.f = f;

        // Resize back the lengths of x, y, z
        state.x.conservativeResize(n);
        state.y.conservativeResize(m);
        state.z.conservativeResize(n);

        // Set the components corresponding to non-trivial variables and constraints
        rows(state.x, inontrivial_variables)   = state.x.segment(0, nn).eval();
        rows(state.y, inontrivial_constraints) = state.y.segment(0, mn).eval();
        rows(state.z, inontrivial_variables)   = state.z.segment(0, nn).eval();

        // Set the components corresponding to trivial variables and constraints
        rows(state.x, ifixedlowerbounds)   = xtrivial;
        rows(state.y, izeroconstraints).fill(0.0);
        rows(state.z, ifixedlowerbounds).fill(0.0);
    }
}

auto Regularizer::Impl::recover(Vector& dxdp) -> void
{
    // Set the components corresponding to trivial and non-trivial variables
    if(izeroconstraints.size())
    {
        const Index nn = inontrivial_variables.size();
        const Index nt = ifixedlowerbounds.size();
        const Index n = nn + nt;
        dxdp.conservativeResize(n);
        rows(dxdp, inontrivial_variables) = dxdp.segment(0, nn).eval();
        rows(dxdp, ifixedlowerbounds).fill(0.0);
    }
}

Regularizer::Regularizer()
: pimpl(new Impl())
{}

Regularizer::Regularizer(const Regularizer& other)
: pimpl(new Impl(*other.pimpl))
{}

Regularizer::~Regularizer()
{}

auto Regularizer::operator=(Regularizer other) -> Regularizer&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Regularizer::setOptions(const RegularizerOptions& options) -> void
{
    pimpl->options = options;
}

auto Regularizer::regularize(const OptimumState& state, const OptimumProblem& problem) -> void
{
	pimpl->regularize(state, problem);
}

auto Regularizer::regularize(const OptimumState& state, const Vector& a) -> void
{
	pimpl->regularize(state, a);
}

auto Regularizer::regularize(const OptimumState& state) -> void
{
	pimpl->regularize(state);
}

} // namespace Optima
