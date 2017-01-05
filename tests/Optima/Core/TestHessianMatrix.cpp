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

// Eigenx includes
#include <Eigenx/Core.hpp>
using namespace Eigen;

// Optima includes
#include <Optima/Core/HessianMatrix.hpp>
using namespace Optima;

TEST_CASE("Testing HessianMatrix when in Diagonal mode")
{
    VectorXd diag = ones(10);
    HessianMatrix H(diag);

    CHECK(identity(10, 10).isApprox(H.convert()));
}

TEST_CASE("Testing HessianMatrix when in Dense mode")
{
    MatrixXd dense = random(10, 10);
    HessianMatrix H(dense);

    CHECK(dense.isApprox(H.convert()));
}
