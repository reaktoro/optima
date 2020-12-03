// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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

#include "EchelonizerW.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/EchelonizerExtended.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct EchelonizerW::Impl
{
    /// The number of columns in Ax and Jx
    const MasterDims dims;

    /// The matrix W = [Ax Ap; Jx Jp]
    Matrix W;

    /// The workspace matrix for S = [Sbn Sbp]
    Matrix S;

    /// The echelonizer of matrix Wx = [Ax; Jx]
    EchelonizerExtended echelonizer;

    Impl(const MasterDims& dims)
    : dims(dims)
    {
        S.resize(dims.nw, dims.nx + dims.np);
        W.resize(dims.nw, dims.nx + dims.np);
    }

    auto initialize(MatrixView Ax, MatrixView Ap) -> void
    {
        const auto [nx, np, ny, nz, nw, nt] = dims;

        using std::max;

        assert(nx > 0);
        assert(Ax.rows() == ny || ny == 0);
        assert(Ax.cols() == nx || ny == 0);
        assert(Ap.rows() == ny || ny == 0 || np == 0);
        assert(Ap.cols() == np || ny == 0 || np == 0);

        // TODO: Implement a sort of memoization here to avoid echelonization of same Ax.
        // If same as last time, instead of creating a new echelonizer, we call echelonizer.initialize(Ax)
        // where EchelonizerExtended::initialize should figure out if same. Careful with echelon form that has
        // been contaminated with round off errors (because there has been many basic swaps already).
        echelonizer = EchelonizerExtended(Ax);

        if(Ax.size()) W. topLeftCorner(ny, nx) = Ax;
        if(Ap.size()) W.topRightCorner(ny, np) = Ap;
    }

    auto update(MatrixView Ax, MatrixView Ap, MatrixView Jx, MatrixView Jp, VectorView weights) -> void
    {
        initialize(Ax, Ap);
        update(Jx, Jp, weights);
    }

    auto update(MatrixView Jx, MatrixView Jp, VectorView weights) -> void
    {
        const auto [nx, np, ny, nz, nw, nt] = dims;

        assert( echelonizer.R().rows() );

        assert(Jx.rows() == nz || nz == 0);
        assert(Jx.cols() == nx || nz == 0);
        assert(Jp.rows() == nz || nz == 0 || np == 0);
        assert(Jp.cols() == np || nz == 0 || np == 0);

        assert( nx == weights.rows() );

        auto Wx = W.leftCols(nx);
        auto Wp = W.rightCols(np);

        if(Jx.size()) Wx.bottomRows(nz) = Jx;
        if(Jp.size()) Wp.bottomRows(nz) = Jp;

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

    auto asMatrixViewW() const -> MatrixViewW
    {
        const auto [nx, np, ny, nz, nw, nt] = dims;

        const auto Wx = W.leftCols(nx);
        const auto Wp = W.rightCols(np);
        const auto Ax = Wx.topRows(ny);
        const auto Ap = Wp.topRows(ny);
        const auto Jx = Wx.bottomRows(nz);
        const auto Jp = Wp.bottomRows(nz);

        return {Wx, Wp, Ax, Ap, Jx, Jp};
    }

    auto asMatrixViewRWQ() const -> MatrixViewRWQ
    {
        const auto [nx, np, ny, nz, nw, nt] = dims;

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

EchelonizerW::EchelonizerW(const MasterDims& dims)
: pimpl(new Impl(dims))
{}

EchelonizerW::EchelonizerW(const EchelonizerW& other)
: pimpl(new Impl(*other.pimpl))
{}

EchelonizerW::~EchelonizerW()
{}

auto EchelonizerW::initialize(MatrixView Ax, MatrixView Ap) -> void
{
    pimpl->initialize(Ax, Ap);
}

auto EchelonizerW::update(MatrixView Ax, MatrixView Ap, MatrixView Jx, MatrixView Jp, VectorView weights) -> void
{
    pimpl->update(Ax, Ap, Jx, Jp, weights);
}

auto EchelonizerW::update(MatrixView Jx, MatrixView Jp, VectorView weights) -> void
{
    pimpl->update(Jx, Jp, weights);
}

auto EchelonizerW::dims() const -> MasterDims
{
    return pimpl->dims;
}

auto EchelonizerW::W() const -> MatrixViewW
{
    return pimpl->asMatrixViewW();
}

auto EchelonizerW::RWQ() const -> MatrixViewRWQ
{
    return pimpl->asMatrixViewRWQ();
}

} // namespace Optima
