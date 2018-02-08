// Optima is a C++ library for numerical sol of linear and nonlinear programing problems.
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

#include <catch.hpp>

// Eigen includes
#include <eigen3/Eigen/LU>
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
    VectorXd s = M.fullPivLu().solve(r);                                    \
    std::cout << "M = \n" << M << std::endl;                                \
    std::cout << "r         = " << tr(r) << std::endl;                           \
    std::cout << "step      = " << tr(step) << std::endl;                        \
    std::cout << "step(lu)  = " << tr(s) << std::endl;      \
    std::cout << "dx        = " << tr(step.head(n)) << std::endl;                        \
    std::cout << "dx(lu)    = " << tr(s.head(n)) << std::endl;      \
    std::cout << "dy        = " << tr(step.segment(n, m)) << std::endl;                        \
    std::cout << "dy(lu)    = " << tr(s.segment(n, m)) << std::endl;      \
    std::cout << "dz        = " << tr(step.segment(n+m, n)) << std::endl;                        \
    std::cout << "dz(lu)    = " << tr(s.segment(n+m, n)) << std::endl;      \
    std::cout << "dw        = " << tr(step.tail(n)) << std::endl;                        \
    std::cout << "dw(lu)    = " << tr(s.tail(n)) << std::endl;      \
    std::cout << "res       = " << tr(res) << std::endl;                         \
    std::cout << "res(lu)   = " << tr(M*s - r) << std::endl;                         \
}                                                                           \

//#undef PRINT_STATE
//#define PRINT_STATE

TEST_CASE("Testing OptimumStepper")
{
//    std::srand(std::time(0));
//    Index n = 60;
//    Index m = 20;
//    Index n = 6;
//    Index m = 3;
    Index n = 2;
    Index m = 1;
    Index t = 3*n + m;

    MatrixXd A = random(m, n);
    MatrixXd H = random(n, n);
    VectorXd g = random(n);
    VectorXd b = random(m);
    VectorXd x = abs(random(n));
    VectorXd y = random(m);
    VectorXd z = abs(random(n));
    VectorXd w = -abs(random(n));
    VectorXd l;
    VectorXd u;

//    MatrixXd A = ones(m, n);
//    MatrixXd H = zeros(n, n);
//    VectorXd g = ones(n);
//    VectorXd a = ones(m);
//    VectorXd x = ones(n);
//    VectorXd y = ones(m);
//    VectorXd z = ones(n);

    OptimumOptions options;

    VectorXd zbar, wbar, lbar, ubar;

    MatrixXd M;

    VectorXd r;

    auto compute_step = [&]()
    {
        OptimumStructure structure(n);
        structure.A = A;
        if(l.size()) structure.withLowerBounds();
        if(u.size()) structure.withUpperBounds();

        OptimumParams params(structure);
        params.b() = b;
        params.xlower() = l;
        params.xupper() = u;

        OptimumState state;
        state.x = x;
        state.y = y;
        state.z = z;
        state.w = w;

        ObjectiveState f;
        f.grad = g;
        f.hessian = H;

        OptimumStepper stepper(structure);
        stepper.setOptions(options);
        stepper.decompose(params, state, f);
        stepper.solve(params, state, f);

        return VectorXd(stepper.step());
    };

    auto compute_residual = [&](VectorXdConstRef step)
    {
        zbar = l.size() ? z : zeros(n);
        wbar = u.size() ? w : zeros(n);
        lbar = l.size() ? VectorXd(x - l) : ones(n);
        ubar = u.size() ? VectorXd(x - u) : ones(n);

        M = zeros(t, t);
        M.topRows(n) << H, tr(A), -identity(n, n), -identity(n, n);
        M.middleRows(n, m).leftCols(n) = A;
        M.middleRows(n + m, n).leftCols(n).diagonal() = zbar;
        M.middleRows(n + m, n).middleCols(n + m, n).diagonal() = lbar;
        M.bottomRows(n).leftCols(n).diagonal() = wbar;
        M.bottomRows(n).rightCols(n).diagonal() = ubar;

        r = zeros(t);
        r.head(n) = -(g + tr(A)*y - zbar - wbar);
        r.segment(n, m) = -(A*x - b);
        r.segment(n + m, n) = -(lbar % zbar - options.mu);
        r.tail(n) = -(ubar % wbar - options.mu);

        return M*step - r;
    };

//    SECTION("When there are no lower/upper bounds.")
//    {
//        VectorXd step = compute_step();
//        VectorXd res = compute_residual(step);
//
//        PRINT_STATE;
//
//        REQUIRE(norm(res)/norm(r) == Approx(0.0));
//    }
//
//    SECTION("When there are lower/upper bounds, but all variables are stable.")
//    {
//        l = zeros(n);
//        u = ones(n);
//        z.fill( options.mu);
//        w.fill(-options.mu);
//
//        VectorXd step = compute_step();
//        VectorXd res = compute_residual(step);
//
//        PRINT_STATE;
//
//        REQUIRE(norm(res)/norm(r) == Approx(0.0));
//    }

    SECTION("When the last `m = nrows(A)` variables are lower unstable.")
    {
        l = zeros(n);
        u = ones(n);
        z.tail(m).fill(1.0);
        x.tail(m).fill(options.mu);

        VectorXd step = compute_step();
        VectorXd res = compute_residual(step);

        PRINT_STATE;

        REQUIRE(norm(res)/norm(r) == Approx(0.0));
    }

//    SECTION("When the last `m = nrows(A)` variables are upper unstable.")
//    {
//        l = zeros(n);
//        u = ones(n);
//        w.tail(m).fill(1.0);
//        x.tail(m).fill(1.0 - options.mu);
//
//        VectorXd step = compute_step();
//        VectorXd res = compute_residual(step);
//
//        PRINT_STATE;
//
//        REQUIRE(norm(res)/norm(r) == Approx(0.0));
//    }
//
//    SECTION("When the last `m = nrows(A)` variables are lower unstable and Huu has large diagonal entries.")
//    {
//        l = zeros(n);
//        u = ones(n);
//        z.tail(m).fill(1.0);
//        x.tail(m).fill(options.mu);
//        H.bottomRightCorner(m, m).diagonal().fill(1e8);
//
//        VectorXd step = compute_step();
//        VectorXd res = compute_residual(step);
//
//        PRINT_STATE;
//
//        REQUIRE(norm(res)/norm(r) == Approx(0.0));
//    }
//
//    SECTION("When the saddle point problem corresponds to a linear programming problem.")
//    {
//        l = zeros(n);
//        g = abs(g);
//        z.tail(n - m).fill(1.0);
//        z.head(m).fill(options.mu);
//        x = options.mu/z;
//        H = zeros(n, n);
//
//        VectorXd step = compute_step();
//        VectorXd res = compute_residual(step);
//
//        PRINT_STATE;
//
//        REQUIRE(norm(res)/norm(r) == Approx(0.0));
//    }
}
