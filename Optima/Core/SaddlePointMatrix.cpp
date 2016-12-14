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

#include "SaddlePointMatrix.hpp"

namespace Optima {

auto SaddlePointMatrix::convert() const -> Matrix
{
    assert(valid());
    const Index n = H.rows();
    const Index m = A.rows();
    const Index p = Z.rows();
    const Index t = n + m + p;
    Matrix res = zeros(t, t);
    res.topLeftCorner(n, n).diagonal() = H;
    res.middleRows(n, m).leftCols(n) = A;
    res.middleCols(n, m).topRows(n) = -tr(A);
    if(p > 0)
    {
        res.topRightCorner(n, n).diagonal() = -ones(n);
        res.bottomLeftCorner(n, n).diagonal() = Z;
        res.bottomRightCorner(n, n).diagonal() = X;
    }
    return res;
}

auto SaddlePointMatrix::valid() const -> bool
{
    if(H.rows() != A.cols()) return false;
    if(Z.rows() && Z.rows() != H.rows()) return false;
    if(Z.rows() != X.rows()) return false;
    return true;
}

auto SaddlePointVector::convert() const -> Vector
{
    assert(valid());
    const Index n = x.rows();
    const Index m = y.rows();
    const Index p = z.rows();
    const Index t = n + m + p;
    Vector res(t);
    res.head(n) = x;
    res.segment(n, m) = y;
    res.tail(p) = z;
    return res;
}

auto SaddlePointVector::valid() const -> bool
{
    if(z.rows() && z.rows() != x.rows()) return false;
    return true;
}


auto SaddlePointMatrixCanonical::convert() const -> Matrix
{
    assert(valid());

    const Index nb = Gb.rows();
    const Index ns = Gs.rows();
    const Index nu = Gu.rows();
    const Index m  = Bb.rows();
    const Index pb = Eb.rows();
    const Index ps = Es.rows();
    const Index pu = Eu.rows();
    const Index n  = nb + ns + nu;
    const Index p  = pb + ps + pu;
    const Index t  = n + m + p;

    Matrix res = zeros(t, t);

    auto G = res.topLeftCorner(n, n).diagonal();
    auto B = res.middleRows(n, m).leftCols(n);
    auto BT = res.middleCols(n, m).topRows(n);
    auto ETR = res.topRightCorner(n, n).diagonal();
    auto EBL = res.bottomLeftCorner(n, n).diagonal();
    auto EBR = res.bottomRightCorner(n, n).diagonal();

    if(nb) G.topRows(nb) = Gb;
    if(ns) G.middleRows(nb, ns) = Gs;
    if(nu) G.bottomRows(nu) = Gu;

    if(pb) ETR.topRows(nb) = Eb;
    if(ps) ETR.middleRows(nb, ns) = Es;
    if(pu) ETR.bottomRows(nu) = Eu;

    if(pb) EBL.topRows(nb) = Eb;
    if(ps) EBL.middleRows(nb, ns) = Es;
    if(pu) EBL.bottomRows(nu) = Eu;

    if(pb) EBR.topRows(nb) = Eb;
    if(ps) EBR.middleRows(nb, ns) = Es;
    if(pu) EBR.bottomRows(nu) = Eu;

    if(nb) B.leftCols(nb) = diag(Bb);
    if(ns) B.middleCols(nb, ns) = Bs;
    if(nu) B.rightCols(nu) = Bu;

    if(nb) BT.topRows(nb).diagonal() = Bb;
    if(ns) BT.middleRows(nb, ns) = tr(Bs);
    if(nu) BT.bottomRows(nu) = tr(Bu);

    return res;
}

auto SaddlePointMatrixCanonical::valid() const -> bool
{
    if(Gb.rows() != Bb.rows()) return false;
    if(Bs.rows() && Bs.rows() != Bb.rows()) return false;
    if(Bu.rows() && Bu.rows() != Bb.rows()) return false;
    if(Bs.rows() && Bs.cols() != Gs.rows()) return false;
    if(Bu.rows() && Bu.cols() != Gu.rows()) return false;
    if(Eb.rows() && Eb.rows() != Gb.rows()) return false;
    if(Es.rows() && Es.rows() != Gs.rows()) return false;
    if(Eu.rows() && Eu.rows() != Gu.rows()) return false;
    return true;
}

auto SaddlePointVectorCanonical::convert() const -> Vector
{
    assert(valid());

    const Index nb = xb.rows();
    const Index ns = xs.rows();
    const Index nu = xu.rows();
    const Index m  = y.rows();
    const Index pb = zb.rows();
    const Index ps = zs.rows();
    const Index pu = zu.rows();
    const Index n  = nb + ns + nu;
    const Index p  = pb + ps + pu;
    const Index t  = n + m + p;

    Vector res(t);

    auto a = res.topRows(n);
    auto c = res.bottomRows(n);

    if(nb) a.head(nb) = xb;
    if(ns) a.segment(nb, ns) = xs;
    if(nu) a.tail(nu) = xu;

    res.middleRows(n, m) = y;

    if(pb) c.head(nb) = zb;
    if(ps) c.segment(nb, ns) = zs;
    if(pu) c.tail(nu) = zu;

    return res;
}

auto SaddlePointVectorCanonical::valid() const -> bool
{
    if(zb.rows() && zb.rows() != xb.rows()) return false;
    if(zs.rows() && zs.rows() != xs.rows()) return false;
    if(zu.rows() && zu.rows() != xu.rows()) return false;
    return true;
}

} // namespace Optima
