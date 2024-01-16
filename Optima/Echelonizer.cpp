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

#include "Echelonizer.hpp"

// Eigen includes
#include <Eigen/Dense>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

/// Return true if matrices A and B are identical.
auto identical(const MatrixView& A, const MatrixView& B)
{
    return A.rows() == B.rows() && A.cols() == B.cols() && A == B;
}

struct Echelonizer::Impl
{
    /// The full-pivoting LU decomposition of A so that P*A*Q = L*U;
    Eigen::FullPivLU<Matrix> lu;

    /// The matrix A being echelonized.
    Matrix A;

    /// The rank of matrix A.
    Index rankA;

    /// The current echelonizer matrix R such that RAQ = C = [I S].
    Matrix R;

    /// The current matrix S in the canonical form C = [I S].
    Matrix S;

    /// The permutation matrix P.
    Indices P;

    /// The transpose of the permutation matrix P.
    Indices Ptr;

    /// The permutation matrix Q.
    Indices Q;

    /// The auxiliary permutation matrix Q.
    Indices Qaux;

    /// The inverse permutation of the new ordering of the variables
    Indices inv_ordering;

    /// The matrix M used in the swap operation.
    Vector M;

    /// The permutation matrix Kb used in the weighted update method.
    PermutationMatrix Kb;

    /// The permutation matrix Kn used in the weighted update method.
    PermutationMatrix Kn;

    /// The backup matrix R used to reset this object to a state with non-accumulated round-off errors.
    Matrix R0;

    /// The backup matrix S used to reset this object to a state with non-accumulated round-off errors.
    Matrix S0;

    /// The backup permutation matrix Q used to reset this object to a state with non-accumulated round-off errors.
    Indices Q0;

    /// The threshold used to compare numbers in S matrix.
    const double threshold = 1e-8;

    /// The number used for eliminating round-off errors during cleanup procedure.
    /// This is computed as 10**[1 + ceil(log10(maxAij))], where maxAij is the
    /// inf norm of matrix A. For each entry in R and S, we add sigma and
    /// subtract sigma, so that residual round-off errors are eliminated.
    double sigma;

    /// Return the number of basic variables, which is also the the rank of matrix A.
    auto numBasicVariables() -> Index
    {
        // Check if max pivot is very small
        if(lu.maxPivot() < 10*std::numeric_limits<double>::epsilon())
        {
            const auto previous_threshold = lu.threshold();
            lu.setThreshold(1.0); // In this case, set threshold to 1, to effectively obtain an absolute comparion instead of relative
            const auto r = lu.rank();
            lu.setThreshold(previous_threshold);
            return r;
        }
        else return lu.rank();
    }

    /// Compute the canonical matrix of the given matrix.
    auto compute(MatrixView Anew) -> void
    {
        // Avoid echelonization if Anew is A
        if(identical(Anew, A))
            return;

        A = Anew;

        // The number of rows and columns of A
        const auto m = A.rows();
        const auto n = A.cols();

        /// Initialize the current ordering of the variables
        inv_ordering = indices(n);

        // Compute the full-pivoting LU of A so that P*A*Q = L*U
        lu.compute(A);

        // Update the rank of matrix A
        rankA = numBasicVariables();

        // The number of basic and non-basic columns of A.
        const auto nb = rankA;
        const auto nn = n - rankA;

        // Get the LU factors of matrix A
        const auto Lbb = lu.matrixLU().topLeftCorner(nb, nb).triangularView<Eigen::UnitLower>();
        const auto U   = lu.matrixLU().rightCols(n).triangularView<Eigen::Upper>();
        const auto Ubb = lu.matrixLU().topLeftCorner(nb, nb).triangularView<Eigen::Upper>();
        const auto Ubn = lu.matrixLU().topRightCorner(nb, nn);

        // Set the permutation matrices P and Q
        P = lu.permutationP().indices().cast<Index>();
        Q = lu.permutationQ().indices().cast<Index>();

        // Initialize the permutation matrix Q(aux)
        Qaux = Q;

        Ptr.resize(m);
        Ptr(P) = indices(m);

        // Calculate the regularizer matrix R
        R = P.asPermutation();
        R.topRows(nb) = Lbb.solve(R.topRows(nb)); // [L] = 6x5, [R] = 6x6, [Lbb] = 5x5
        R.topRows(nb) = Ubb.solve(R.topRows(nb));

        // Calculate matrix S
        S = Ubn;
        S = Ubb.solve(S);

        // Initialize the permutation matrices Kb and Kn
        Kb.setIdentity(nb);
        Kn.setIdentity(nn);

        // Compute sigma for given matrix A
        sigma = A.size() ? A.cwiseAbs().maxCoeff() : 0.0;
        sigma = A.size() ? std::pow(10, 1 + std::ceil(std::log10(sigma))) : 0.0;

        // Set the backup matrices R0, S0, Q0 for resetting purposes
        R0 = R;
        S0 = S;
        Q0 = Q;
    }

    /// Swap a basic variable by a non-basic variable.
    auto updateWithSwapBasicVariable(Index ib, Index in) -> void
    {
        // The number of basic and non-basic columns of A.
        const auto nb = rankA;
        const auto nn = A.cols() - rankA;

        // Check if ib < rank(A)
        assert(ib < nb &&
            "Could not swap basic and non-basic variables. "
                "Expecting an index of basic variable below `r`, where `r = rank(A)`.");

        // Check if in < n - rank(A)
        assert(in < nn &&
            "Could not swap basic and non-basic variables. "
                "Expecting an index of non-basic variable below `n - r`, where `r = rank(A)`.");

        // Check if S(ib, in) is large enough
        assert(std::abs(S(ib, in)) > threshold &&
            "Could not swap basic and non-basic variables. "
                "Expecting a non-basic variable with large enough S(ib, in) pivot.");

        // Initialize the matrix M
        M = S.col(in);

        // Auxiliary variables
        const auto m = S.rows();
        const auto aux = 1.0/S(ib, in);

        // Update the echelonizer matrix R (only its `r` upper rows, where `r = rank(A)`)
        R.row(ib) *= aux;
        for(auto i = 0; i < m; ++i)
            if(i != ib) R.row(i) -= S(i, in) * R.row(ib);

        // Update matrix S
        S.row(ib) *= aux;
        for(auto i = 0; i < m; ++i)
            if(i != ib) S.row(i) -= S(i, in) * S.row(ib);
        S.col(in) = -M*aux;
        S(ib, in) = aux;

        // Update the permutation matrix Q
        std::swap(Q[ib], Q[m + in]);
    }

    /// Update the existing canonical form with given priority weights for the columns.
    auto updateWithPriorityWeights(VectorView w) -> void
    {
        // Assert there are as many weights as there are variables
        assert(w.rows() == lu.cols() &&
            "Could not update the canonical form."
                "Mismatch number of variables and given priority weights.");

        // The number of basic and non-basic columns of A.
        const auto nb = rankA;
        const auto nn = A.cols() - rankA;

        // The upper part of R corresponding to linearly independent rows of A
        auto Rb = R.topRows(nb);

        // The indices of the basic and non-basic variables
        auto ibasic = Q.head(nb);
        auto inonbasic = Q.tail(nn);

        // Find the non-basic variable with maximum proportional weight with respect to a basic variable
        auto find_nonbasic_candidate = [&](Index i, Index& j)
        {
            j = 0; double max = -infinity();
            double tmp = 0.0;
            for(Index k = 0; k < nn; ++k) {
                if(std::abs(S(i, k)) <= threshold) continue;
                tmp = w[inonbasic[k]];
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
                updateWithSwapBasicVariable(i, j);
        }

        // Sort the basic variables in descend order of weights
        std::sort(Kb.indices().data(), Kb.indices().data() + nb,
            [&](Index l, Index r) { return w[ibasic[l]] > w[ibasic[r]]; });

        // Sort the non-basic variables in descend order of weights
        std::sort(Kn.indices().data(), Kn.indices().data() + nn,
            [&](Index l, Index r) { return w[inonbasic[l]] > w[inonbasic[r]]; });

        // Rearrange the rows of S based on the new order of basic variables
        Kb.transpose().applyThisOnTheLeft(S);

        // Rearrange the columns of S based on the new order of non-basic variables
        Kn.applyThisOnTheRight(S);

        // Rearrange the top `nb` rows of R based on the new order of basic variables
        Kb.transpose().applyThisOnTheLeft(Rb);

        // Rearrange the permutation matrix Q based on the new order of basic variables
        Kb.transpose().applyThisOnTheLeft(ibasic);

        // Rearrange the permutation matrix Q based on the new order of non-basic variables
        Kn.transpose().applyThisOnTheLeft(inonbasic);
    }

    /// Reset to the canonical matrix form computed initially.
    auto reset() -> void
    {
        R = R0;
        S = S0;
        Q = Q0;
    }

    /// Update the ordering of the basic and non-basic variables,
    auto updateOrdering(IndicesView Kb, IndicesView Kn) -> void
    {
        const auto n  = Q.rows();
        const auto nb = S.rows();
        const auto nn = n - nb;

        assert(nb == Kb.size());
        assert(nn == Kn.size());

        // The top nb rows of R, since its remaining rows correspond to linearly dependent rows in A
        auto Rt = R.topRows(nb);

        // The indices of the basic and non-basic variables
        auto ibasic = Q.head(nb);
        auto inonbasic = Q.tail(nn);

        // Rearrange the rows of S based on the new order of basic variables
        Kb.asPermutation().transpose().applyThisOnTheLeft(S);

        // Rearrange the columns of S based on the new order of non-basic variables
        Kn.asPermutation().applyThisOnTheRight(S);

        // Rearrange the top `nb` rows of R based on the new order of basic variables
        Kb.asPermutation().transpose().applyThisOnTheLeft(Rt);

        // Rearrange the permutation matrix Q based on the new order of basic variables
        Kb.asPermutation().transpose().applyThisOnTheLeft(ibasic);

        // Rearrange the permutation matrix Q based on the new order of non-basic variables
        Kn.asPermutation().transpose().applyThisOnTheLeft(inonbasic);
    }

    /// Perform a cleanup procedure to remove residual round-off errors from the canonical form.
    auto cleanResidualRoundoffErrors() -> void
    {
        S.array() += sigma;
        S.array() -= sigma;

        R.array() += sigma;
        R.array() -= sigma;
    }
};

Echelonizer::Echelonizer()
: pimpl(new Impl())
{}

Echelonizer::Echelonizer(MatrixView A)
: pimpl(new Impl())
{
    compute(A);
}

Echelonizer::Echelonizer(const Echelonizer& other)
: pimpl(new Impl(*other.pimpl))
{}

Echelonizer::~Echelonizer()
{}

auto Echelonizer::operator=(Echelonizer other) -> Echelonizer&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Echelonizer::numVariables() const -> Index
{
    return pimpl->lu.cols();
}

auto Echelonizer::numEquations() const -> Index
{
    return pimpl->lu.rows();
}

auto Echelonizer::numBasicVariables() const -> Index
{
    return pimpl->rankA;
}

auto Echelonizer::numNonBasicVariables() const -> Index
{
    return numVariables() - numBasicVariables();
}

auto Echelonizer::S() const -> MatrixView
{
    return pimpl->S;
}

auto Echelonizer::R() const -> MatrixView
{
    return pimpl->R;
}

auto Echelonizer::Q() const -> IndicesView
{
    return pimpl->Q;
}

auto Echelonizer::C() const -> Matrix
{
    const Index m  = numEquations();
    const Index n  = numVariables();
    const Index nb = numBasicVariables();
    Matrix res = zeros(m, n);
    res.topRows(nb) << identity(nb, nb), S();
    return res;
}

auto Echelonizer::indicesEquations() const -> IndicesView
{
    return pimpl->Ptr;
}

auto Echelonizer::indicesBasicVariables() const -> IndicesView
{
    return Q().head(numBasicVariables());
}

auto Echelonizer::indicesNonBasicVariables() const -> IndicesView
{
    return Q().tail(numNonBasicVariables());
}

auto Echelonizer::compute(MatrixView A) -> void
{
    pimpl->compute(A);
}

auto Echelonizer::updateWithSwapBasicVariable(Index ibasic, Index inonbasic) -> void
{
    pimpl->updateWithSwapBasicVariable(ibasic, inonbasic);
}

auto Echelonizer::updateWithPriorityWeights(VectorView weights) -> void
{
    pimpl->updateWithPriorityWeights(weights);
}

auto Echelonizer::updateOrdering(IndicesView Kb, IndicesView Kn) -> void
{
    pimpl->updateOrdering(Kb, Kn);
}

auto Echelonizer::reset() -> void
{
    pimpl->reset();
}

auto Echelonizer::cleanResidualRoundoffErrors() -> void
{
    pimpl->cleanResidualRoundoffErrors();
}

} // namespace Optima
