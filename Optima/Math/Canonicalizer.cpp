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
    Eigen::FullPivLU<Matrix> lu;

    /// The matrix `S` in the canonical form `C = [I S]`.
    Matrix S;

    /// The permutation matrix `P`.
    PermutationMatrix P;

    /// The permutation matrix `Q`.
    PermutationMatrix Q;

    /// The canonicalizer matrix `R`.
    Matrix R;

    /// The inverse of the canonicalizer matrix `R`.
    Matrix Rinv;

    /// The priority weights used to update the canonical form.
    Vector w;

    /// The matrix `M` used in the swap operation.
    Vector M;

    /// The permutation matrix `Kb` used in the weighted update method.
    Indices bswaps;

    /// The permutation matrix `Kn` used in the weighted update method.
    Indices nswaps;
};

Canonicalizer::Canonicalizer()
: pimpl(new Impl())
{}

Canonicalizer::Canonicalizer(const Matrix& A)
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

auto Canonicalizer::rows() const -> Index
{
    return S().rows();
}

auto Canonicalizer::cols() const -> Index
{
    return Q().rows();
}

auto Canonicalizer::S() const -> const Matrix&
{
	return pimpl->S;
}

auto Canonicalizer::R() const -> const Matrix&
{
	return pimpl->R;
}

auto Canonicalizer::Rinv() const -> const Matrix&
{
	return pimpl->Rinv;
}

auto Canonicalizer::Q() const -> const PermutationMatrix&
{
	return pimpl->Q;
}

auto Canonicalizer::C() const -> Matrix
{
    const Index m = rows();
    const Index n = cols();
    Matrix res(m, n);
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

auto Canonicalizer::compute(const Matrix& A) -> void
{
    // Alias to implementation members
    auto& P      = pimpl->P;
    auto& Q      = pimpl->Q;
    auto& S      = pimpl->S;
    auto& R      = pimpl->R;
    auto& lu     = pimpl->lu;
    auto& Rinv   = pimpl->Rinv;
    auto& bswaps = pimpl->bswaps;
    auto& nswaps = pimpl->nswaps;

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

	// Calculate the inverse of the regularizer matrix R
	Rinv = P.transpose();
	Rinv.conservativeResize(m, r);
	Rinv = Rinv * Lbb;
	Rinv = Rinv * Ubb;

	// Calculate matrix S
	S = Ubn;
	S = Ubb.solve(S);

	// Initialize the vectors of basic and non-basic swaps
	bswaps = indices(r);
	nswaps = indices(n - r);
}

auto Canonicalizer::swap(Index ib, Index in) -> void
{
	// Alias to implementation members
	auto& M    = pimpl->M;
	auto& Q    = pimpl->Q.indices();
	auto& S    = pimpl->S;
	auto& R    = pimpl->R;
	auto& Rinv = pimpl->Rinv;

	// Check if S(ib, in) is different than zero
	Assert(S(ib, in), "Could not swap basic and non-basic components.",
		"Expecting a non-basic component with non-zero pivot.");

	// Initialize the matrix M
	M = S.col(in);

	// Auxiliary variables
	const Index m = S.rows();
	const double aux = 1.0/S(ib, in);

	// Updadte the canonicalizer matrix R
	R.row(ib) *= aux;
	for(Index i = 0; i < m; ++i)
		if(i != ib) R.row(i) -= S(i, in) * R.row(ib);

	// Updadte the inverse of the canonicalizer matrix R
	Rinv.col(ib) = Rinv * S.col(in);

	// Updadte matrix S
	S.row(ib) *= aux;
	for(Index i = 0; i < m; ++i)
		if(i != ib) S.row(i) -= S(i, in) * S.row(ib);
	S.col(in) = -M*aux;
	S(ib, in) = aux;

	// Update the permutation matrix Q
	std::swap(Q[ib], Q[m + in]);
}

auto Canonicalizer::update(const Vector& weights) -> void
{
    update(weights, {});
}

auto Canonicalizer::update(const Vector& weights, const Indices& fixed) -> void
{
	// Auxiliary references to member data
	auto& Q      = pimpl->Q;
	auto& S      = pimpl->S;
	auto& R      = pimpl->R;
	auto& Rinv   = pimpl->Rinv;
	auto& w      = pimpl->w;
    auto& bswaps = pimpl->bswaps;
    auto& nswaps = pimpl->nswaps;

	// Auxiliary variables
	const Index m  = rows();
	const Index n  = cols();
	const Index nb = m;
	const Index nn = n - nb;
	const Index nf = fixed.size();

	// The indices of the basic and non-basic variables
	auto ibasic = Q.indices().head(nb);
	auto inonbasic = Q.indices().tail(nn);

    // Update the internal weights to update the canonical form of A.
    if(weights.rows()) w.noalias() = abs(weights); else w = ones(n);

    // Set weights of fixed variables to zero to prevent them from becoming basic variables
    Optima::rows(w, fixed).fill(0.0);

	// The weights of the non-basic components
	auto wn = Optima::rows(w, inonbasic);

	// Swap basic and non-basic components when the latter has higher weight
	if(nn > 0) for(Index i = 0; i < nb; ++i)
	{
        Index j;
		const double wi = std::abs(w[ibasic[i]]);
		const double wj = abs(wn % tr(S.row(i))).maxCoeff(&j);
		if(wi < wj)
			swap(i, j);
	}

    // Set weights of fixed variables to decreasing negative values to move them to the back of the list
    Optima::rows(w, fixed) = -linspace(nf, 1, nf);

	// Sort the basic components in descend order of weights
	std::sort(bswaps.data(), bswaps.data() + bswaps.size(),
		[&](Index l, Index r) { return std::abs(w[ibasic[l]]) > std::abs(w[ibasic[r]]); });

	// Sort the non-basic components in descend order of weights
	std::sort(nswaps.data(), nswaps.data() + nswaps.size(),
		[&](Index l, Index r) { return std::abs(w[inonbasic[l]]) > std::abs(w[inonbasic[r]]); });

	// Rearrange the rows of S based on the new order of basic components
    Index i = 0;
    while(i < nb) {
        if(i != bswaps[i]) {
            S.row(i).swap(S.row(bswaps[i]));
            R.row(i).swap(R.row(bswaps[i]));
            Rinv.col(i).swap(Rinv.col(bswaps[i]));
            std::swap(ibasic[i], ibasic[bswaps[i]]);
            std::swap(bswaps[i], bswaps[bswaps[i]]);
        }
        else ++i;
    }

	// Rearrange the columns of S based on the new order of non-basic components
    Index j = 0;
    while(j < nn) {
        if(j != nswaps[j]) {
            S.col(j).swap(S.col(nswaps[j]));
            std::swap(inonbasic[j], inonbasic[nswaps[j]]);
            std::swap(nswaps[j], nswaps[nswaps[j]]);
        }
        else ++j;
    }
}

} // namespace Optima
