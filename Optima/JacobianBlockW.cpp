// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
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

#include "JacobianBlockW.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/ExtendedCanonicalizer.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct JacobianBlockW::Impl
{
    const Index nx;  ///< The number of columns in Ax and Jx
    const Index np;  ///< The number of columns in Ap and Jp
    const Index ny;  ///< The number of rows in Ax and Ap
    const Index nz;  ///< The number of rows in Jx and Jp
    const Index nw;  ///< The number of rows in W = [Ax Ap; Jx Jp]
    Matrix W;        ///< The matrix W = [Ax Ap; Jx Jp]
    Matrix S;        ///< The workspace matrix for S = [Sbn Sbp]
    ExtendedCanonicalizer canonicalizer; ///< The canonicalizer of matrix Wx = [Ax; Jx]

    Impl(Index nx, Index np, Index ny, Index nz, MatrixConstRef Ax, MatrixConstRef Ap)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz), canonicalizer(Ax)
    {
        assert( ny == Ax.rows() );
        assert( ny == Ap.rows() );

        S.resize(nw, nx + np);
        W.resize(nw, nx + np);

        auto Wx = W.leftCols(nx);
        auto Wp = W.rightCols(np);

        Wx.topRows(ny) = Ax;
        Wp.topRows(ny) = Ap;
    }

    auto update(MatrixConstRef Jx, MatrixConstRef Jp, VectorConstRef weights) -> void
    {
        assert( nz == Jx.rows() );
        assert( nz == Jp.rows() );
        assert( nx == Jx.cols() );
        assert( np == Jp.cols() );
        assert( nx == weights.rows() );

        auto Wx = W.leftCols(nx);
        auto Wp = W.rightCols(np);

        Wx.bottomRows(nz) = Jx;
        Wp.bottomRows(nz) = Jp;

        // weights = min(x - xlower, xupper - x);
        // weights = weights.array().isInf().select(abs(x), weights); // replace inf distance with abs(x)
        // weights.noalias() = abs(weights);
        // weights = (weights.array() > 0.0).select(weights, -1.0); // set negative priority weights for variables on the bounds
        canonicalizer.updateWithPriorityWeights(Jx, weights);
        canonicalizer.cleanResidualRoundoffErrors();

        const auto nb = canonicalizer.numBasicVariables();
        const auto nn = canonicalizer.numNonBasicVariables();

        const auto Rb = canonicalizer.R().topRows(nb);

        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        Sbn = canonicalizer.S();
        Sbp = Rb * Wp;

        cleanResidualRoundoffErrors(Sbp);
    }

    auto canonicalForm() const -> CanonicalForm
    {
        const auto nb = canonicalizer.numBasicVariables();
        const auto nn = canonicalizer.numNonBasicVariables();

        const auto R   = canonicalizer.R();
        const auto Sbn = S.topLeftCorner(nb, nn);
        const auto Sbp = S.topRightCorner(nb, np);
        const auto jbn = canonicalizer.Q();
        const auto jb  = jbn.head(nb);
        const auto jn  = jbn.tail(nn);

        return {R, Sbn, Sbp, jb, jn};
    }

    auto Ax() const -> MatrixConstRef { return W.topLeftCorner(ny, nx); }
    auto Ap() const -> MatrixConstRef { return W.topRightCorner(ny, np); }
    auto Jx() const -> MatrixConstRef { return W.bottomLeftCorner(nz, nx); }
    auto Jp() const -> MatrixConstRef { return W.bottomRightCorner(nz, np); }
    auto Wx() const -> MatrixConstRef { return W.leftCols(nx); }
    auto Wp() const -> MatrixConstRef { return W.rightCols(np); }
};

JacobianBlockW::JacobianBlockW(Index nx, Index np, Index ny, Index nz, MatrixConstRef Ax, MatrixConstRef Ap)
: pimpl(new Impl(nx, np, ny, nz, Ax, Ap)),
  Ax(pimpl->Ax()),
  Ap(pimpl->Ap()),
  Jx(pimpl->Jx()),
  Jp(pimpl->Jp()),
  Wx(pimpl->Wx()),
  Wp(pimpl->Wp())
{}

JacobianBlockW::JacobianBlockW(const JacobianBlockW& other)
: pimpl(new Impl(*other.pimpl)),
  Ax(pimpl->Ax()),
  Ap(pimpl->Ap()),
  Jx(pimpl->Jx()),
  Jp(pimpl->Jp()),
  Wx(pimpl->Wx()),
  Wp(pimpl->Wp())
{}

JacobianBlockW::~JacobianBlockW()
{}

auto JacobianBlockW::update(MatrixConstRef Jx, MatrixConstRef Jp, VectorConstRef weights) -> void
{
    pimpl->update(Jx, Jp, weights);
}

auto JacobianBlockW::canonicalForm() const -> CanonicalForm
{
    return pimpl->canonicalForm();
}

} // namespace Optima
