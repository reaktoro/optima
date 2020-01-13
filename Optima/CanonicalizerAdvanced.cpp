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

#include "CanonicalizerAdvanced.hpp"

// Optima includes
#include <Optima/Canonicalizer.hpp>

namespace Optima {

struct CanonicalizerAdvanced::Impl
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

    /// Construct a default CanonicalizerAdvanced::Impl object
    Impl()
    {
    }

    /// Construct a CanonicalizerAdvanced::Impl object with given matrix A
    Impl(MatrixConstRef A, MatrixConstRef J)
    {
        compute(A, J);
    }

    /// Compute the canonical matrix with given upper and lower matrix blocks \eq{A} and \eq{J}.
    auto compute(MatrixConstRef A, MatrixConstRef J) -> void
    {
        canonicalizerA.compute(A);
        auto weights = ones(A.cols());
        update(J, weights);
    }

    /// Update the canonical form with given matrix J and priority weights for the variables.
    auto update(MatrixConstRef J, VectorConstRef weights) -> void
    {
        canonicalizerA.updateWithPriorityWeights(weights);

        if(J.size() == 0)
        {
            R = canonicalizerA.R();
            S = canonicalizerA.S();
            Q = canonicalizerA.Q();
        }
        else
        {
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
        }
    }
};

CanonicalizerAdvanced::CanonicalizerAdvanced()
: pimpl(new Impl())
{
}

CanonicalizerAdvanced::CanonicalizerAdvanced(MatrixConstRef A, MatrixConstRef J)
: pimpl(new Impl(A, J))
{
}

CanonicalizerAdvanced::CanonicalizerAdvanced(const CanonicalizerAdvanced& other)
: pimpl(new Impl(*other.pimpl))
{}

CanonicalizerAdvanced::~CanonicalizerAdvanced()
{}

auto CanonicalizerAdvanced::operator=(CanonicalizerAdvanced other) -> CanonicalizerAdvanced&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto CanonicalizerAdvanced::numVariables() const -> Index
{
    return pimpl->Q.rows();
}

auto CanonicalizerAdvanced::numEquations() const -> Index
{
    return pimpl->canonicalizerA.numEquations() + pimpl->canonicalizerJ.numEquations();
}

auto CanonicalizerAdvanced::numBasicVariables() const -> Index
{
    return pimpl->S.rows();
}

auto CanonicalizerAdvanced::numNonBasicVariables() const -> Index
{
    return numVariables() - numBasicVariables();
}

auto CanonicalizerAdvanced::S() const -> MatrixConstRef
{
    return pimpl->S;
}

auto CanonicalizerAdvanced::R() const -> MatrixConstRef
{
    return pimpl->R;
}

auto CanonicalizerAdvanced::Q() const -> IndicesConstRef
{
    return pimpl->Q;
}

auto CanonicalizerAdvanced::C() const -> Matrix
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

auto CanonicalizerAdvanced::indicesBasicVariables() const -> IndicesConstRef
{
    const auto nb = numBasicVariables();
    return pimpl->Q.head(nb);
}

auto CanonicalizerAdvanced::indicesNonBasicVariables() const -> IndicesConstRef
{
    const auto nn = numNonBasicVariables();
    return pimpl->Q.tail(nn);
}

auto CanonicalizerAdvanced::compute(MatrixConstRef A, MatrixConstRef J) -> void
{
    pimpl->compute(A, J);
}

auto CanonicalizerAdvanced::update(MatrixConstRef J, VectorConstRef weights) -> void
{
    pimpl->update(J, weights);
}

} // namespace Optima
