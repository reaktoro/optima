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
#include <Optima/Optima.hpp>
using namespace Optima;

TEST_CASE("Testing HessianMatrix when in Zero mode")
{
    HessianMatrix H;
    H.zero(10);

    CHECK(zeros(10, 10).isApprox(H));
    CHECK(zeros(10, 10).isApprox(Matrix(H)));
}

TEST_CASE("Testing HessianMatrix when in Diagonal mode")
{
    HessianMatrix H;
    H.diagonal(10).fill(1.0);

    CHECK(identity(10, 10).isApprox(H));
    CHECK(identity(10, 10).isApprox(Matrix(H)));

    Vector r1 = random(3);
    Vector r2 = random(5);
    H.blocks(2);
    H.block(0).diagonal(3) = r1;
    H.block(1).diagonal(5) = r2;

    Matrix A = zeros(8, 8);
    A.diagonal().head(3) = r1;
    A.diagonal().tail(5) = r2;

    CHECK(A.isApprox(H));
    CHECK(A.isApprox(Matrix(H)));
}

TEST_CASE("Testing HessianMatrix when in Dense mode")
{
    HessianMatrix H;
    H.dense(10).fill(1.0);

    CHECK(ones(10, 10).isApprox(H));
    CHECK(ones(10, 10).isApprox(Matrix(H)));

    Matrix r1 = random(3, 3);
    Matrix r2 = random(5, 5);
    H.blocks(2);
    H.block(0).dense(3) = r1;
    H.block(1).dense(5) = r2;

    Matrix A = zeros(8, 8);
    A.topLeftCorner(3, 3)     = r1;
    A.bottomRightCorner(5, 5) = r2;

    CHECK(A.isApprox(H));
    CHECK(A.isApprox(Matrix(H)));
}

TEST_CASE("Testing HessianMatrix when in EigenDecomp mode")
{
    HessianMatrix H;
    auto& eigen = H.eigendecomposition(10);
    eigen.eigenvalues = random(10);
    eigen.eigenvectors = diag(2.0 * ones(10));
    eigen.eigenvectorsinv = diag(0.5 * ones(10));

    Matrix A = eigen.eigenvectors * diag(eigen.eigenvalues) * eigen.eigenvectorsinv;

    CHECK(A.isApprox(H));
    CHECK(A.isApprox(Matrix(H)));
}
