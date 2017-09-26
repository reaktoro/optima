// Optima is a C++ library for numerical sol of linear and nonlinear programing problems.
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

// Eigen includes
#include <Eigenx/LU.hpp>
using namespace Eigen;

// Optima includes
#include <Optima/Core/OptimumOptions.hpp>
#include <Optima/Core/OptimumParams.hpp>
#include <Optima/Core/OptimumState.hpp>
#include <Optima/Core/OptimumStepper.hpp>
#include <Optima/Core/OptimumStructure.hpp>
#include <Optima/Math/Matrix.hpp>
using namespace Optima;

#define PRINT_STATE                                                         \
{                                                                           \
    std::cout << std::setprecision(10); \
    VectorXd s = M.fullPivLu().solve(r);                                    \
    std::cout << "M = \n" << M << std::endl;                                \
    std::cout << "r         = " << tr(r) << std::endl;                           \
    std::cout << "step      = " << tr(step) << std::endl;                        \
    std::cout << "step(lu)  = " << tr(s) << std::endl;      \
    std::cout << "res       = " << tr(res) << std::endl;                         \
    std::cout << "res(lu)   = " << tr(M*s - r) << std::endl;                         \
}                                                                           \

TEST_CASE("Testing OptimumStepper")
{
    std::srand(std::time(0));
    Index n = 60;
    Index m = 20;
//    Index n = 6;
//    Index m = 3;
    Index t = 2*n + m;

    MatrixXd A = random(m, n);
    MatrixXd H = random(n, n);
    VectorXd g = random(n);
    VectorXd a = random(m);
    VectorXd x = abs(random(n));
    VectorXd y = random(m);
    VectorXd z = abs(random(n));

//    MatrixXd A = ones(m, n);
//    MatrixXd H = zeros(n, n);
//    VectorXd g = ones(n);
//    VectorXd a = ones(m);
//    VectorXd x = ones(n);
//    VectorXd y = ones(m);
//    VectorXd z = ones(n);

    OptimumOptions options;

    auto assemble_matrix = [&]()
    {
        MatrixXd M = zeros(t, t);
        M.topLeftCorner(n, n) = H;
        M.topRows(n).middleCols(n, m) = tr(A);
        M.topRightCorner(n, n).diagonal().fill(-1.0);
        M.middleRows(n, m).leftCols(n) = A;
        M.bottomLeftCorner(n, n).diagonal() = z;
        M.bottomRightCorner(n, n).diagonal() = x;
        return M;
    };

    auto assemble_vector = [&]()
    {
        VectorXd r = zeros(t);
        r.head(n) = -(g + tr(A)*y - z);
        r.segment(n, m) = -(A*x - a);
        r.tail(n) = -(x % z - options.mu);
        return r;
    };

    auto compute_step = [&]()
    {
        OptimumStructure structure;
        structure.n = n;
        structure.A = A;

        OptimumParams params;
        params.a = a;
        params.xlower = zeros(n);

        OptimumState state;
        state.x = x;
        state.y = y;
        state.z = z;

        ObjectiveState f;
        f.grad = g;
        f.hessian = H;

        OptimumStepper stepper;
        stepper.setOptions(options);
        stepper.initialize(structure);
        stepper.decompose(params, state, f);
        stepper.solve(params, state, f);

        return VectorXd(stepper.step());
    };

    SUBCASE("When all variables are stable.")
    {
        z.noalias() = 1e-8 * x;
        MatrixXd M = assemble_matrix();
        MatrixXd r = assemble_vector();
        VectorXd step = compute_step();
        VectorXd res = M*step - r;

//        PRINT_STATE;

        CHECK(norm(res)/norm(r) == approx(0.0));
    }

    SUBCASE("When the last `m = nrows(A)` variables are unstable.")
    {
        z.tail(m).fill(1.0);
        x.tail(m).fill(options.mu);

        MatrixXd M = assemble_matrix();
        MatrixXd r = assemble_vector();
        VectorXd step = compute_step();
        VectorXd res = M*step - r;

//        PRINT_STATE;

        CHECK(norm(res)/norm(r) == approx(0.0));
    }

    SUBCASE("When the last `m = nrows(A)` variables are unstable and Huu has large diagonal entries.")
    {
        z.tail(m).fill(1.0);
        x.tail(m).fill(options.mu);
        H.bottomRightCorner(m, m).diagonal().fill(1e8);

        MatrixXd M = assemble_matrix();
        MatrixXd r = assemble_vector();
        VectorXd step = compute_step();
        VectorXd res = M*step - r;

//        PRINT_STATE;

        CHECK(norm(res)/norm(r) == approx(0.0));
    }

    SUBCASE("When the saddle point problem corresponds to a linear programming problem.")
    {
        g = abs(g);
        z.tail(n - m).fill(1.0);
        z.head(m).fill(options.mu);
        x = options.mu/z;
        H = zeros(n, n);

        MatrixXd M = assemble_matrix();
        MatrixXd r = assemble_vector();
        VectorXd step = compute_step();
        VectorXd res = M*step - r;

//        PRINT_STATE;

        CHECK(norm(res)/norm(r) == approx(0.0));
    }
}
