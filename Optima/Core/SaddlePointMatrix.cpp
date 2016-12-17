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
    const Index t = 2*n + m;

    Matrix res = zeros(t, t);
    res.topLeftCorner(n, n).diagonal()     = H;
    res.middleRows(n, m).leftCols(n)       = A;
    res.middleCols(n, m).topRows(n)        = -tr(A);
    res.topRightCorner(n, n).diagonal()    = -ones(n);
    res.bottomLeftCorner(n, n).diagonal()  = Z;
    res.bottomRightCorner(n, n).diagonal() = X;

    return res;
}

auto SaddlePointMatrix::valid() const -> bool
{
    if(A.cols() != H.rows()) return false;
    if(Z.rows() != H.rows()) return false;
    if(X.rows() != H.rows()) return false;
    return true;
}

auto SaddlePointVector::convert() const -> Vector
{
    assert(valid());

    const Index n = x.rows();
    const Index m = y.rows();
    const Index t = 2*n + m;
    Vector res(t);
    res << x, y, z;
    return res;
}

auto SaddlePointVector::valid() const -> bool
{
    if(z.rows() != x.rows()) return false;
    if(y.rows() == 0) return false;
    return true;
}


auto SaddlePointMatrixCanonical::convert() const -> Matrix
{
    assert(valid());

    const Index m  = nb;
    const Index n  = nb + ns + nu;
    const Index t  = 2*n + m;

    Matrix res = zeros(t, t);
    res.topLeftCorner(n, n).diagonal()           = G;
    res.middleRows(n, m).leftCols(nb).diagonal() = Bb;
    res.middleRows(n, m).rightCols(nn)           = Bn;
    res.middleCols(n, m).topRows(nb).diagonal()  = Bb;
    res.middleCols(n, m).bottomRows(nn)          = tr(Bn);
    res.topRightCorner(n, n).diagonal()          = E;
    res.bottomLeftCorner(n, n).diagonal()        = E;
    res.bottomRightCorner(n, n).diagonal()       = E;

    return res;
}

auto SaddlePointMatrixCanonical::valid() const -> bool
{
    const Index n = G.rows();
    if(nb + ns + nu != n) return false;
    if(nb + nn != n) return false;
    if(nb != Index(Bb.rows())) return false;
    if(nn != Index(Bn.cols())) return false;
    if(E.rows() != G.rows()) return false;
    return true;
}

} // namespace Optima
