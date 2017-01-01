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

#include <doctest/doctest.hpp>

// Optima includes
#include <Optima/Math/Matrix.hpp>
using namespace Optima;

TEST_CASE("Testing Matrix")
{
    Matrix A = random(10,10);
    Indices irows1 = {1, 2};
    Indices irows2 = {1};
    auto B = rows(A, irows1);
    auto C = rows(rows(A, irows1), irows2);
    auto D = cols(A, irows1);
    auto E = cols(cols(A, irows1), irows2);
    auto F = submatrix(A, irows1, irows1);

    CHECK(A.row(1).isApprox(B.row(0)));
    CHECK(A.row(2).isApprox(B.row(1)));
    CHECK(A.row(2).isApprox(C.row(0)));
    CHECK(A.col(1).isApprox(D.col(0)));
    CHECK(A.col(2).isApprox(D.col(1)));
    CHECK(A.col(2).isApprox(E.col(0)));
    CHECK(A(1, 1) == F(0, 0));
    CHECK(A(1, 2) == F(0, 1));
    CHECK(A(2, 1) == F(1, 0));
    CHECK(A(2, 2) == F(1, 1));
}

TEST_CASE("Testing const Matrix")
{
    const Matrix A = random(10,10);
    Indices irows1 = {1, 2};
    Indices irows2 = {1};
    auto B = rows(A, irows1);
    auto C = rows(rows(A, irows1), irows2);
    auto D = cols(A, irows1);
    auto E = cols(cols(A, irows1), irows2);
    auto F = submatrix(A, irows1, irows1);

    CHECK(A.row(1).isApprox(B.row(0)));
    CHECK(A.row(2).isApprox(B.row(1)));
    CHECK(A.row(2).isApprox(C.row(0)));
    CHECK(A.col(1).isApprox(D.col(0)));
    CHECK(A.col(2).isApprox(D.col(1)));
    CHECK(A.col(2).isApprox(E.col(0)));
    CHECK(A(1, 1) == F(0, 0));
    CHECK(A(1, 2) == F(0, 1));
    CHECK(A(2, 1) == F(1, 0));
    CHECK(A(2, 2) == F(1, 1));
}


