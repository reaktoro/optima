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

#include "MatrixRWQ.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/EchelonizerExtended.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct MatrixRWQ::Impl
{
    const Index nx;  ///< The number of columns in Ax and Jx
    const Index np;  ///< The number of columns in Ap and Jp
    const Index ny;  ///< The number of rows in Ax and Ap
    const Index nz;  ///< The number of rows in Jx and Jp
    const Index nw;  ///< The number of rows in W = [Ax Ap; Jx Jp]
    Matrix W;        ///< The matrix W = [Ax Ap; Jx Jp]
    Matrix S;        ///< The workspace matrix for S = [Sbn Sbp]

    /// The echelonizer of matrix Wx = [Ax; Jx]
    EchelonizerExtended echelonizer;

    Impl(Index nx, Index np, Index ny, Index nz, MatrixConstRef Ax, MatrixConstRef Ap)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz), echelonizer(Ax)
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
        echelonizer.updateWithPriorityWeights(Jx, weights);
        echelonizer.cleanResidualRoundoffErrors();

        const auto nb = echelonizer.numBasicVariables();
        const auto nn = echelonizer.numNonBasicVariables();

        const auto Rb = echelonizer.R().topRows(nb);

        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        Sbn = echelonizer.S();
        Sbp = Rb * Wp;

        cleanResidualRoundoffErrors(Sbp);
    }

    auto view() const -> MatrixViewRWQ
    {
        const auto nb = echelonizer.numBasicVariables();
        const auto nn = echelonizer.numNonBasicVariables();

        const auto Rb   = echelonizer.R().topRows(nb);
        const auto Sbn = S.topLeftCorner(nb, nn);
        const auto Sbp = S.topRightCorner(nb, np);
        const auto jbn = echelonizer.Q();
        const auto jb  = jbn.head(nb);
        const auto jn  = jbn.tail(nn);

        return {Rb, Sbn, Sbp, jb, jn};
    }
};

MatrixRWQ::MatrixRWQ(Index nx, Index np, Index ny, Index nz, MatrixConstRef Ax, MatrixConstRef Ap)
: pimpl(new Impl(nx, np, ny, nz, Ax, Ap))
{}

MatrixRWQ::MatrixRWQ(const MatrixRWQ& other)
: pimpl(new Impl(*other.pimpl))
{}

MatrixRWQ::~MatrixRWQ()
{}

auto MatrixRWQ::update(MatrixConstRef Jx, MatrixConstRef Jp, VectorConstRef weights) -> void
{
    pimpl->update(Jx, Jp, weights);
}

auto MatrixRWQ::view() const -> MatrixViewRWQ
{
    return pimpl->view();
}

} // namespace Optima
