// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Jlan Leal
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

#include "ExtendedCanonicalizer.hpp"

// Optima includes
#include <Optima/Canonicalizer.hpp>

namespace Optima {

struct ExtendedCanonicalizer::Impl
{
    /// The canonicalizer for matrix A
    Canonicalizer canonicalizerA;

    /// The canonicalizer for matrix J
    Canonicalizer canonicalizerJ;

    /// The matrix R in the canonicalization R*[A; J]*Q = [I S]
    Matrix R;

    /// The matrix S in the canonicalization R*[A; J]*Q = [I S]
    Matrix S;

    /// The permutation matrix Q in the canonicalization R*[A; J]*Q = [I S]
    Indices Q;

    /// The permutation matrix `Kb` used in the update method with priority weights.
    PermutationMatrix Kb;

    /// The permutation matrix `Kn` used in the update method with priority weights.
    PermutationMatrix Kn;

    /// Construct a default ExtendedCanonicalizer::Impl object
    Impl()
    {
    }

    /// Construct a ExtendedCanonicalizer::Impl object with given matrix A
    Impl(MatrixConstRef A)
    {
        // Initialize the canonicalizer for A (wait until J is provided to initialize canonicalizerJ)
        canonicalizerA.compute(A);
        R = canonicalizerA.R();
        S = canonicalizerA.S();
        Q = canonicalizerA.Q();
    }

    /// Update the canonical form with given variable matrix J in W = [A; J] and priority weights for the variables.
    auto updateWithPriorityWeights(MatrixConstRef J, VectorConstRef weights) -> void
    {
        canonicalizerA.updateWithPriorityWeights(weights);

        if(J.size() == 0)
        {
            R = canonicalizerA.R();
            S = canonicalizerA.S();
            Q = canonicalizerA.Q();
        }

        const auto& RA = canonicalizerA.R();
        const auto& SA = canonicalizerA.S();
        const auto& QA = canonicalizerA.Q();

        const auto n = canonicalizerA.numVariables();
        const auto nbA = canonicalizerA.numBasicVariables();
        const auto nnA = canonicalizerA.numNonBasicVariables();

        const auto mA = canonicalizerA.numEquations();
        const auto mJ = J.rows();
        const auto m = mA + mJ;

        Matrix J12 = J * QA.asPermutation();
        auto J1 = J12.leftCols(nbA);
        auto J2 = J12.rightCols(nnA);
        J2 -= J1 * SA;

        Vector w = weights(QA.tail(nnA));  // w has the weights only for non-basic variables wrt A
        canonicalizerJ.compute(J2);
        canonicalizerJ.updateWithPriorityWeights(w);

        const auto nbJ = canonicalizerJ.numBasicVariables();
        const auto nnJ = canonicalizerJ.numNonBasicVariables();

        const auto RJ = canonicalizerJ.R();
        const auto SJ = canonicalizerJ.S();
        const auto QJ = canonicalizerJ.Q();

        Q.resize(n);
        Q.head(nbA) = QA.head(nbA);
        Q.tail(nnA) = QA.tail(nnA)(QJ);

        Matrix SA12 = SA;
        QJ.asPermutation().applyThisOnTheRight(SA12);
        auto SA1 = SA12.leftCols(nbJ);
        auto SA2 = SA12.rightCols(nnJ);
        SA2 -= SA1 * SJ;

        S.resize(nbA + nbJ, nnJ);
        S.topRows(nbA) = SA2;
        S.bottomRows(nbJ) = SJ;

        const auto RAt = RA.topRows(nbA);
        const auto RAb = RA.bottomRows(mA - nbA);

        const auto RJt = RJ.topRows(nbJ);
        const auto RJb = RJ.bottomRows(mJ - nbJ);

        R.resize(m, m);

        auto Rt = R.topRows(nbA + nbJ);
        auto Rb = R.bottomRows(m - nbA - nbJ);

        Rt.topLeftCorner(nbA, mA) = RAt + SA1*RJt*J1*RAt;
        Rb.topLeftCorner(mA - nbA, mA) = RAb;

        Rt.topRightCorner(nbA, mJ) = -SA1*RJt;
        Rb.topRightCorner(mA - nbA, mJ).fill(0.0);

        Rt.bottomLeftCorner(nbJ, mA) = -RJt*J1*RAt;
        Rb.bottomLeftCorner(mJ - nbJ, mA) = -RJb*J1*RAt;

        Rt.bottomRightCorner(nbJ, mJ) = RJt;
        Rb.bottomRightCorner(mJ - nbJ, mJ) = RJb;

        //---------------------------------------------------------------------------
        // Start sorting of basic and non-basic variables according to their weights
        //---------------------------------------------------------------------------

        // Initialize the permutation matrices Kb and Kn
        auto nb = nbA + nbJ;
        auto nn = n - nb;

        Kb.setIdentity(nb);
        Kn.setIdentity(nn);

        // The indices of the basic and non-basic variables
        auto ibasic = Q.head(nb);
        auto inonbasic = Q.tail(nn);

        // Sort the basic variables in descend order of weights
        std::sort(Kb.indices().data(), Kb.indices().data() + nb,
            [&](Index l, Index r) { return weights[ibasic[l]] > weights[ibasic[r]]; });

        // Sort the non-basic variables in descend order of weights
        std::sort(Kn.indices().data(), Kn.indices().data() + nn,
            [&](Index l, Index r) { return weights[inonbasic[l]] > weights[inonbasic[r]]; });

        // Rearrange the rows of S based on the new order of basic variables
        Kb.transpose().applyThisOnTheLeft(S);

        // Rearrange the columns of S based on the new order of non-basic variables
        Kn.applyThisOnTheRight(S);

        // Rearrange the top `nb` rows of R based on the new order of basic variables
        Kb.transpose().applyThisOnTheLeft(Rt);

        // Rearrange the permutation matrix Q based on the new order of basic variables
        Kb.transpose().applyThisOnTheLeft(ibasic);

        // Rearrange the permutation matrix Q based on the new order of non-basic variables
        Kn.transpose().applyThisOnTheLeft(inonbasic);
    }

    /// Update the ordering of the basic and non-basic variables,
    auto updateOrdering(IndicesConstRef Kb, IndicesConstRef Kn) -> void
    {
        const auto n  = Q.rows();
        const auto nb = S.rows();
        const auto nn = n - nb;

        assert(nb == Kb.size());
        assert(nn == Kn.size());

        // The top nb rows of R, since its remaining rows correspond to linearly dependent rows in [A; J]
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
};

ExtendedCanonicalizer::ExtendedCanonicalizer()
: pimpl(new Impl())
{
}

ExtendedCanonicalizer::ExtendedCanonicalizer(MatrixConstRef A)
: pimpl(new Impl(A))
{
}

ExtendedCanonicalizer::ExtendedCanonicalizer(const ExtendedCanonicalizer& other)
: pimpl(new Impl(*other.pimpl))
{}

ExtendedCanonicalizer::~ExtendedCanonicalizer()
{}

auto ExtendedCanonicalizer::operator=(ExtendedCanonicalizer other) -> ExtendedCanonicalizer&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto ExtendedCanonicalizer::numVariables() const -> Index
{
    return pimpl->Q.rows();
}

auto ExtendedCanonicalizer::numEquations() const -> Index
{
    return pimpl->canonicalizerA.numEquations() + pimpl->canonicalizerJ.numEquations();
}

auto ExtendedCanonicalizer::numBasicVariables() const -> Index
{
    return pimpl->S.rows();
}

auto ExtendedCanonicalizer::numNonBasicVariables() const -> Index
{
    return numVariables() - numBasicVariables();
}

auto ExtendedCanonicalizer::S() const -> MatrixConstRef
{
    return pimpl->S;
}

auto ExtendedCanonicalizer::R() const -> MatrixConstRef
{
    return pimpl->R;
}

auto ExtendedCanonicalizer::Q() const -> IndicesConstRef
{
    return pimpl->Q;
}

auto ExtendedCanonicalizer::C() const -> Matrix
{
    const auto n = numVariables();
    const auto m = numEquations();
    const auto nb = numBasicVariables();
    const auto nn = numNonBasicVariables();
    Matrix res = zeros(m, n);
    res.topLeftCorner(nb, nb).setIdentity(nb, nb);
    res.topRightCorner(nb, nn) = S();
    return res;
}

auto ExtendedCanonicalizer::indicesBasicVariables() const -> IndicesConstRef
{
    const auto nb = numBasicVariables();
    return pimpl->Q.head(nb);
}

auto ExtendedCanonicalizer::indicesNonBasicVariables() const -> IndicesConstRef
{
    const auto nn = numNonBasicVariables();
    return pimpl->Q.tail(nn);
}

auto ExtendedCanonicalizer::updateWithPriorityWeights(MatrixConstRef J, VectorConstRef weights) -> void
{
    pimpl->updateWithPriorityWeights(J, weights);
}

auto ExtendedCanonicalizer::updateOrdering(IndicesConstRef Kb, IndicesConstRef Kn) -> void
{
    pimpl->updateOrdering(Kb, Kn);
}

} // namespace Optima
