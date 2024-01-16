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

#include "EchelonizerExtended.hpp"

// C++ includes
#include <cmath>

// Optima includes
#include <Optima/Echelonizer.hpp>
#include <Optima/Exception.hpp>

namespace Optima {

struct EchelonizerExtended::Impl
{
    /// The echelonizer for matrix A
    Echelonizer echelonizerA;

    /// The echelonizer for matrix J
    Echelonizer echelonizerJ;

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

    /// The number used for eliminating round-off errors during cleanup procedure.
    /// This is computed as 10**[1 + ceil(log10(maxAij))], where maxAij is the
    /// inf norm of matrix A. For each entry in R and S, we add sigma and
    /// subtract sigma, so that residual round-off errors are eliminated.
    double sigma;

    /// Construct a default EchelonizerExtended::Impl object
    Impl()
    {
    }

    /// Construct a EchelonizerExtended::Impl object with given matrix A
    Impl(MatrixView A)
    {
        // Initialize the echelonizer for A (wait until J is provided to initialize echelonizerJ)
        echelonizerA.compute(A);
        R = echelonizerA.R();
        S = echelonizerA.S();
        Q = echelonizerA.Q();

        // Compute sigma for given matrix A
        sigma = A.size() ? A.cwiseAbs().maxCoeff() : 0.0;
        sigma = A.size() ? std::pow(10, 1 + std::ceil(std::log10(sigma))) : 0.0; // TODO: In the future, consider a contribution from J to determine sigma (or find an alternative approach to remove round-off errors.)
    }

    /// Update the canonical form with given variable matrix J in W = [A; J] and priority weights for the variables.
    auto updateWithPriorityWeights(MatrixView J, VectorView weights) -> void
    {
        echelonizerA.updateWithPriorityWeights(weights);

        if(J.size() == 0)
        {
            R = echelonizerA.R();
            S = echelonizerA.S();
            Q = echelonizerA.Q();
            return;
        }

        // FIXME: Investigate why EchelonizerExtended is not accurate when nz > 5 and fix it.

        const auto& RA = echelonizerA.R();
        const auto& SA = echelonizerA.S();
        const auto& QA = echelonizerA.Q();

        const auto n = echelonizerA.numVariables();
        const auto nbA = echelonizerA.numBasicVariables();
        const auto nnA = echelonizerA.numNonBasicVariables();

        const auto mA = echelonizerA.numEquations();
        const auto mJ = J.rows();
        const auto m = mA + mJ;

        Matrix J12 = J * QA.asPermutation();
        auto J1 = J12.leftCols(nbA);
        auto J2 = J12.rightCols(nnA);
        J2 -= J1 * SA;

        Vector w = weights(QA.tail(nnA));  // w has the weights only for non-basic variables wrt A
        echelonizerJ.compute(J2);
        echelonizerJ.updateWithPriorityWeights(w);

        const auto nbJ = echelonizerJ.numBasicVariables();
        const auto nnJ = echelonizerJ.numNonBasicVariables();

        // TODO: When testing with nx = 30, ny = 20, nz = 5, and two linearly dependent rows,
        // EchelonizerExtended does not produce accurate C = [I S] matrix when performing R*[A; J]*Q.
        // While there are ~1e-14 errors on the very left part of I, which is a reasonable approximation for zeros,
        // for the right side part of I, the last 5 columns, we see ~1e-6, which is not acceptable.
        warning(nbA + nbJ != std::min(m, n), "EchelonizerExtended guarantees accuracy at the moment only when A and J have no linearly dependent rows.");

        const auto RJ = echelonizerJ.R();
        const auto SJ = echelonizerJ.S();
        const auto QJ = echelonizerJ.Q();

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
    auto updateOrdering(IndicesView Kb, IndicesView Kn) -> void
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

    /// Perform a cleanup procedure to remove residual round-off errors from the canonical form.
    auto cleanResidualRoundoffErrors() -> void
    {
        S.array() += sigma;
        S.array() -= sigma;

        R.array() += sigma;
        R.array() -= sigma;
    }
};

EchelonizerExtended::EchelonizerExtended()
: pimpl(new Impl())
{
}

EchelonizerExtended::EchelonizerExtended(MatrixView A)
: pimpl(new Impl(A))
{
}

EchelonizerExtended::EchelonizerExtended(const EchelonizerExtended& other)
: pimpl(new Impl(*other.pimpl))
{}

EchelonizerExtended::~EchelonizerExtended()
{}

auto EchelonizerExtended::operator=(EchelonizerExtended other) -> EchelonizerExtended&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto EchelonizerExtended::numVariables() const -> Index
{
    return pimpl->Q.rows();
}

auto EchelonizerExtended::numEquations() const -> Index
{
    return pimpl->echelonizerA.numEquations() + pimpl->echelonizerJ.numEquations();
}

auto EchelonizerExtended::numBasicVariables() const -> Index
{
    return pimpl->S.rows();
}

auto EchelonizerExtended::numNonBasicVariables() const -> Index
{
    return numVariables() - numBasicVariables();
}

auto EchelonizerExtended::S() const -> MatrixView
{
    return pimpl->S;
}

auto EchelonizerExtended::R() const -> MatrixView
{
    return pimpl->R;
}

auto EchelonizerExtended::Q() const -> IndicesView
{
    return pimpl->Q;
}

auto EchelonizerExtended::C() const -> Matrix
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

auto EchelonizerExtended::indicesBasicVariables() const -> IndicesView
{
    const auto nb = numBasicVariables();
    return pimpl->Q.head(nb);
}

auto EchelonizerExtended::indicesNonBasicVariables() const -> IndicesView
{
    const auto nn = numNonBasicVariables();
    return pimpl->Q.tail(nn);
}

auto EchelonizerExtended::updateWithPriorityWeights(MatrixView J, VectorView weights) -> void
{
    pimpl->updateWithPriorityWeights(J, weights);
}

auto EchelonizerExtended::updateOrdering(IndicesView Kb, IndicesView Kn) -> void
{
    pimpl->updateOrdering(Kb, Kn);
}

auto EchelonizerExtended::cleanResidualRoundoffErrors() -> void
{
    pimpl->cleanResidualRoundoffErrors();
}

} // namespace Optima
