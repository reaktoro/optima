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

#include "Canonicalizer.hpp"

// Eigen includes
#include <Optima/Common/Exception.hpp>
#include <Optima/Math/EigenExtern.hpp>
using namespace Eigen;

namespace Optima {

struct Canonicalizer::Impl
{
    /// The full-pivoting LU decomposition of A so that P*A*Q = L*U;
    Eigen::FullPivLU<MatrixXd> lu;

    /// The matrix `S` in the canonical form `C = [I S]`.
    MatrixXd S;

    /// The permutation matrix `P`.
    PermutationMatrix P;

    /// The permutation matrix `Q`.
    PermutationMatrix Q;

    /// The canonicalizer matrix `R`.
    MatrixXd R;

    /// The matrix `M` used in the swap operation.
    VectorXd M;

    /// The permutation matrix `Kb` used in the weighted update method.
    PermutationMatrix Kb;

    /// The permutation matrix `Kn` used in the weighted update method.
	PermutationMatrix Kn;

	/// The number of free (nbx) and fixed (nbf) basic variables.
	Index nbx, nbf;

	/// The number of free (nnx) and fixed (nnf) non-basic variables.
	Index nnx, nnf;

    /// Compute the canonical matrix of the given matrix.
	auto compute(const MatrixXd& A) -> void
	{
	    // The number of rows and columns of A
	    const Index m = A.rows();
	    const Index n = A.cols();

	    // Check if number of columns is greater/equal than number of rows
	    Assert(n >= m, "Could not canonicalize the given matrix.",
	        "The given matrix has more rows than columns.");

	    // Compute the full-pivoting LU of A so that P*A*Q = L*U
	    lu.compute(A);

	    // Get the rank of matrix A
	    const Index r = lu.rank();

	    // Get the LU factors of matrix A
	    const auto Lbb = lu.matrixLU().topLeftCorner(r, r).triangularView<Eigen::UnitLower>();
	    const auto Ubb = lu.matrixLU().topLeftCorner(r, r).triangularView<Eigen::Upper>();
	    const auto Ubn = lu.matrixLU().topRightCorner(r, n - r);

	    // Set the permutation matrices P and Q
	    P = lu.permutationP();
	    Q = lu.permutationQ();

	    // Calculate the regularizer matrix R
	    R = P;
	    R.conservativeResize(r, m);
	    R = Lbb.solve(R);
	    R = Ubb.solve(R);

	    // Calculate matrix S
	    S = Ubn;
	    S = Ubb.solve(S);

	    // Initialize the permutation matrices Kb and Kn
	    Kb.setIdentity(r);
	    Kn.setIdentity(n - r);

	    // Set the number of basic and non-basic variables (fixed and free)
	    nbf = 0; nbx = r;
	    nnf = 0; nnx = n - r;
	}

    /// Swap a basic variable by a non-basic variable.
	auto swapBasicVariable(Index ib, Index in) -> void
	{
	    // Check if S(ib, in) is different than zero
	    Assert(S(ib, in), "Could not swap basic and non-basic variables.",
	        "Expecting a non-basic variable with non-zero pivot.");

	    // Initialize the matrix M
	    M = S.col(in);

	    // Auxiliary variables
	    const Index m = S.rows();
	    const double aux = 1.0/S(ib, in);

	    // Update the canonicalizer matrix R
	    R.row(ib) *= aux;
	    for(Index i = 0; i < m; ++i)
	        if(i != ib) R.row(i) -= S(i, in) * R.row(ib);

	    // Updadte matrix S
	    S.row(ib) *= aux;
	    for(Index i = 0; i < m; ++i)
	        if(i != ib) S.row(i) -= S(i, in) * S.row(ib);
	    S.col(in) = -M*aux;
	    S(ib, in) = aux;

	    // Update the permutation matrix Q
	    std::swap(Q.indices()[ib], Q.indices()[m + in]);
	}

    /// Update the existing canonical form with given priority weights for the columns.
	auto update(const VectorXd& w) -> void
	{
	    // The rank and number of columns of matrix A
	    const Index r = lu.rank();
	    const Index n = lu.cols();

	    // The number of basic and non-basic variables
	    const Index nb = r;
	    const Index nn = n - r;

	    // The indices of the basic and non-basic variables
	    auto ibasic = Q.indices().head(nb);
	    auto inonbasic = Q.indices().tail(nn);

	    // Find the non-basic variable with maximum proportional weight with respect to a basic variable
	    auto find_nonbasic_candidate = [&](Index i, Index& j)
        {
	        j = 0; double max = w[inonbasic[0]] * std::abs(S(i, 0));
	        double tmp = 0.0;
	        for(Index k = 1; k < nn; ++k) {
	            tmp = w[inonbasic[k]] * std::abs(S(i, k));
	            if(tmp > max) {
	                max = tmp;
	                j = k;
	            }
	        }
	        return max;
        };

	    // Check if there are basic variables to be swapped with non-basic variables with higher priority
	    if(nn > 0) for(Index i = 0; i < nb; ++i)
	    {
	        Index j;
	        const double wi = w[ibasic[i]];
	        const double wj = find_nonbasic_candidate(i, j);
	        if(wi < wj)
	            swapBasicVariable(i, j);
	    }

	    // Sort the basic variables in descend order of weights
	    std::sort(Kb.indices().data(), Kb.indices().data() + nb,
	        [&](Index l, Index r) { return w[ibasic[l]] > w[ibasic[r]]; });

	    // Sort the non-basic variables in descend order of weights
	    std::sort(Kn.indices().data(), Kn.indices().data() + nn,
	        [&](Index l, Index r) { return w[inonbasic[l]] > w[inonbasic[r]]; });

	    // Rearrange the rows of S based on the new order of basic variables
	    Kb.applyThisOnTheLeft(S);

	    // Rearrange the columns of S based on the new order of non-basic variables
	    Kn.applyThisOnTheRight(S);

	    // Rearrange the rows of R based on the new order of basic variables
	    Kb.applyThisOnTheLeft(R);

	    // Rearrange the permutation matrix Q based on the new order of basic variables
	    Kb.applyThisOnTheLeft(ibasic);

	    // Rearrange the permutation matrix Q based on the new order of non-basic variables
	    Kn.transpose().applyThisOnTheLeft(inonbasic);

	    // Find the number of fixed basic variables (those with weights below or equal to zero)
	    nbf = 0; while(nbf < nb && w[ibasic[nb - nbf - 1]] <= 0.0) ++nbf;

	    // Find the number of fixed non-basic variables (those with weights below or equal to zero)
	    nnf = 0; while(nnf < nn && w[inonbasic[nn - nnf - 1]] <= 0.0) ++nnf;

	    // Set the number of free basic and non-basic variables
	    nbx = nb - nbf;
	    nnx = nn - nnf;
	}

//    /// Update the existing canonical form with given priority weights for the columns.
//	auto update(const VectorXd& weights, const Indices& fixed) -> void
//	{
//
//	    // Auxiliary variables
//	    const Index m  = S.rows();
//	    const Index n  = Q.rows();
//	    const Index nb = m;
//	    const Index nn = n - nb;
//	    const Index nf = fixed.size();
//
//	    // The indices of the basic and non-basic variables
//	    auto ibasic = Q.indices().head(nb);
//	    auto inonbasic = Q.indices().tail(nn);
//
//	    // Update the internal weights to update the canonical form of A.
//	    w.noalias() = abs(weights);
//
//	    // Set weights of fixed variables to zero to prevent them from becoming basic variables
//	    Eigen::rows(w, fixed).fill(0.0);
//
//	    // The weights of the non-basic variables
//	    auto wn = Eigen::rows(w, inonbasic);
//
//	    // Swap basic and non-basic variables when the latter has higher weight
//	    if(nn > 0) for(Index i = 0; i < nb; ++i)
//	    {
//	        Index j;
//	        const double wi = w[ibasic[i]];
//	        const double wj = abs(wn % tr(S.row(i))).maxCoeff(&j);
//	        if(wi < wj)
//	            swapBasicVariable(i, j);
//	    }
//
//	    // Set weights of fixed variables to decreasing negative values to move them to the back of the list
//	    Eigen::rows(w, fixed) = -VectorXd::LinSpaced(nf, 1, nf);
//
//	    // Sort the basic variables in descend order of weights
//	    std::sort(Kb.indices().data(), Kb.indices().data() + nb,
//	        [&](Index l, Index r) { return w[ibasic[l]] > w[ibasic[r]]; });
//
//	    // Sort the non-basic variables in descend order of weights
//	    std::sort(Kn.indices().data(), Kn.indices().data() + nn,
//	        [&](Index l, Index r) { return w[inonbasic[l]] > w[inonbasic[r]]; });
//
//	    // Rearrange the rows of S based on the new order of basic variables
//	    Kb.applyThisOnTheLeft(S);
//
//	    // Rearrange the columns of S based on the new order of non-basic variables
//	    Kn.applyThisOnTheRight(S);
//
//	    // Rearrange the rows of R based on the new order of basic variables
//	    Kb.applyThisOnTheLeft(R);
//
//	    // Rearrange the columns of inv(R) based on the new order of basic variables
//	    Kb.transpose().applyThisOnTheRight(Rinv);
//
//	    // Rearrange the permutation matrix Q based on the new order of basic and non-basic variables
//	    Kb.applyThisOnTheLeft(ibasic);
//	    Kn.transpose().applyThisOnTheLeft(inonbasic);
//	}
};

Canonicalizer::Canonicalizer()
: pimpl(new Impl())
{}

Canonicalizer::Canonicalizer(const MatrixXd& A)
: pimpl(new Impl())
{
	compute(A);
}

Canonicalizer::Canonicalizer(const Canonicalizer& other)
: pimpl(new Impl(*other.pimpl))
{}

Canonicalizer::~Canonicalizer()
{}

auto Canonicalizer::operator=(Canonicalizer other) -> Canonicalizer&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Canonicalizer::numBasicVariables() const -> Index
{
    return pimpl->nbx + pimpl->nbf;
}

auto Canonicalizer::numNonBasicVariables() const -> Index
{
    return pimpl->nnx + pimpl->nnf;
}

auto Canonicalizer::numFreeBasicVariables() const -> Index
{
    return pimpl->nbx;
}

auto Canonicalizer::numFreeNonBasicVariables() const -> Index
{
    return pimpl->nnx;
}

auto Canonicalizer::numFixedBasicVariables() const -> Index
{
    return pimpl->nbf;
}

auto Canonicalizer::numFixedNonBasicVariables() const -> Index
{
    return pimpl->nnf;
}

auto Canonicalizer::rows() const -> Index
{
    return S().rows();
}

auto Canonicalizer::cols() const -> Index
{
    return Q().rows();
}

auto Canonicalizer::S() const -> const MatrixXd&
{
	return pimpl->S;
}

auto Canonicalizer::R() const -> const MatrixXd&
{
	return pimpl->R;
}

auto Canonicalizer::Q() const -> const PermutationMatrix&
{
	return pimpl->Q;
}

auto Canonicalizer::C() const -> MatrixXd
{
    const Index m = rows();
    const Index n = cols();
    MatrixXd res(m, n);
    res << identity(m, m), S();
    return res;
}

auto Canonicalizer::ili() const -> Indices
{
	PermutationMatrix Ptr = pimpl->P.transpose();
	auto begin = Ptr.indices().data();
	return Indices(begin, begin + rows());
}

auto Canonicalizer::ordering() const -> Indices
{
	auto begin = Q().indices().data();
	return Indices(begin, begin + cols());
}

auto Canonicalizer::ibasic() const -> Indices
{
	auto begin = Q().indices().data();
	return Indices(begin, begin + rows());
}

auto Canonicalizer::inonbasic() const -> Indices
{
	auto begin = Q().indices().data();
	return Indices(begin + rows(), begin + cols());
}

auto Canonicalizer::compute(const MatrixXd& A) -> void
{
    pimpl->compute(A);
}

auto Canonicalizer::swapBasicVariable(Index ibasic, Index inonbasic) -> void
{
    pimpl->swapBasicVariable(ibasic, inonbasic);
}

auto Canonicalizer::update(const VectorXd& weights) -> void
{
    pimpl->update(weights, {});
}

} // namespace Optima
