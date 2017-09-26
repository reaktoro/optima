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

#include "OptimumStepper.hpp"


#include <iostream> // todo check if necessary later

// Eigenx includes
#include <Eigenx/LU.hpp> // todo check if necessary later
using Eigen::placeholders::all;



// Optima includes
#include <Optima/Core/OptimumOptions.hpp>
#include <Optima/Core/OptimumStructure.hpp>
#include <Optima/Core/OptimumParams.hpp>
#include <Optima/Core/OptimumState.hpp>
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Core/SaddlePointSolver.hpp>
using namespace Eigen;

namespace Optima {

struct OptimumStepper::Impl
{
    /// The structure of the optimization problem.
    OptimumStructure structure;

    /// The options for the optimization calculation
    OptimumOptions options;

    /// The solution vector `sol = [dx dy dz dw]`.
    VectorXd solution;

    /// The right-hand side residual vector `res = [rx ry rz rw]`.
    VectorXd residual;

    /// The `A` matrix in the KKT equation.
    MatrixXd A;

    /// The `H` matrix in the KKT equation.
    MatrixXd H;

    /// The `G` matrix in the KKT equation.
    MatrixXd G;

    VectorXd x, z, g;

    /// The KKT solver.
    SaddlePointSolver kkt;

    /// The order of the variables as `x = [x(stable) x(unstable) x(fixed)]`.
    VectorXi iordering;

    /// The number of variables.
    Index n;

    /// The current number of stable, unstable, free, and fixed variables.
    Index ns, nu, nx, nf;

    /// The number of equality constraints.
    Index m;

    /// The total number of variables (x, y, z, w).
    Index t;

    /// Initialize the stepper with the structure of the optimization problem.
    auto initialize(const OptimumStructure& structure) -> void
    {
        // TODO not sure if we need this
        this->structure = structure;

        // Initialize the members related to number of variables and constraints
        n  = structure.n;
        ns = n;
        nu = 0;
        nx = n;
        nf = 0;
        m  = structure.A.rows();
        t  = 2*n + m;

        // Allocate memory for some members
        A = structure.A;
        H = zeros(n, n);
        G = zeros(m, m);
        g = zeros(n);
        residual = zeros(t);
        solution = zeros(t);

        // Initialize the ordering of the variables
        iordering.setLinSpaced(n, 0, n - 1);

        // Initialize the saddle point solver
        kkt.canonicalize(A);
    }

    /// Decompose the KKT matrix equation used to compute the step vectors.
    auto decompose(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        // Auxiliary variables
        const double eps = std::sqrt(options.mu);

        // Update the number of fixed and free variables
        nf = params.ifixed.size();
        nx = n - nf;

        // Partition the variables into free and fixed variables x = [xx xf]
        iordering.tail(nf).swap(iordering(params.ifixed));

        // Partition the free variables into stable and unstable variables xx = [xs xu]
        auto it = std::partition(iordering.data(), iordering.data() + nx,
            [&](Index i) { return state.x[i] > eps || state.z[i] < eps; });

        // Update the number of stable and unstable free variables
        ns = it - iordering.data();
        nu = nx - ns;

        // Ensure the number of stable variables are positive
        assert(ns > 0 && "Could not compute the step."
            "The number of stable variables must be positive."); // TODO Instead of this assert, consider returning a trivial solution with all variables fixed on the bounds

        // Permute the columns of A so that A = [As Au Af]
        iordering.asPermutation().applyThisOnTheRight(A);

        // The columns of A corresponding to unstable variables
        const auto Au = A.middleCols(ns, nu);

        // The variables x arranged in the ordering x = [xs xu xf]
        x.noalias() = state.x(iordering);

        // The variables z arranged in the ordering z = [zs zu zf]
        z.noalias() = state.z(iordering);

        // The indices of stable, unstable, and unstable+fixed variables
        const auto js = iordering.head(ns);
        const auto ju = iordering.segment(ns, nu);

        // Views to the blocks Hss and Huu of the Hessian matrix
        auto Hss = H.topLeftCorner(ns, ns);
        auto Huu = H.bottomRightCorner(nu, nu).diagonal();
        auto Duu = H.col(0).segment(ns, nu);

        std::cout << "H1 = \n" << H << std::endl;

        // Views to segments xs and xu in x = [xs xu]
        const auto xs = x.head(ns);
        const auto xu = x.segment(ns, nu);

        // Views to segments zs and zu in z = [zs zu]
        const auto zs = z.head(ns);
        const auto zu = z.segment(ns, nu);

        // Assemble the matrices Hss and Huu
        Hss.noalias() = f.hessian(js, js);

        std::cout << "H2 = \n" << H << std::endl;

        Huu.noalias() = f.hessian(ju, ju).diagonal();
        Duu.noalias() = Huu.diagonal();

        std::cout << "H3 = \n" << H << std::endl;


        // Calculate Hss' = Hss + inv(Xs)*Zs
        Hss.diagonal() += zs/xs;

        std::cout << "H4 = \n" << H << std::endl;


        // Calculate Huu' = Iuu + Huu*inv(Zu)*Xu
        Duu.noalias() = 1.0 + Duu % xu/zu;

        std::cout << "H5 = \n" << H << std::endl;


        // Calculate Huu'' = inv(Zu)*Xu*inv(Huu')
        Duu.noalias() = xu/(zu % Duu);

        std::cout << "H6 = \n" << H << std::endl;

        // Calculate G = - Au * Huu'' * tr(Au)
        G.noalias() = - Au * diag(Duu) * tr(Au);

        // Update the decomposition of the KKT matrix
        kkt.reorder(iordering);
        kkt.decompose({H, A, G, ns, nu + nf});
    }

    /// Solve the KKT matrix equation.
    auto solve(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        // The indices of stable, unstable, and unstable+fixed variables
        const auto jx = iordering.head(nx); // free variables
        const auto jf = iordering.tail(nf); // fixed variables
        const auto js = jx.head(ns);        // stable free variables
        const auto ju = jx.tail(nu);        // unstable free variables

        // Views to the blocks Hss and Huu of the Hessian matrix
        auto Hss = H.topLeftCorner(ns, ns);
        auto Huu = H.bottomRightCorner(nu, nu).diagonal();
        auto Duu = H.col(0).segment(ns, nu);

        // Views to sub-vectors of the gradient of the objective function
        auto gx = g.head(nx); // free variables
        auto gf = g.tail(nf); // fixed variables

        auto a = residual.head(n);
        auto b = residual.segment(n, m);
        auto c = residual.tail(n);

        auto ax = a.head(nx);
        auto af = a.tail(nf);
        auto as = ax.head(ns);
        auto au = ax.tail(nu);

        auto cx = c.head(nx);
        auto cf = c.tail(nf);
        auto cs = cx.head(ns);
        auto cu = cx.tail(nu);

        auto xx = x.head(nx);
        auto xf = x.tail(nf);
        auto xs = xx.head(ns);
        auto xu = xx.tail(nu);

        auto zx = z.head(nx);
        auto zf = z.tail(nf);
        auto zs = zx.head(ns);
        auto zu = zx.tail(nu);

        auto dx = solution.head(n);
        auto dy = solution.segment(n, m);
        auto dz = solution.tail(n);

        auto dxx = dx.head(nx);
        auto dxf = dx.tail(nf);
        auto dxs = dxx.head(ns);
        auto dxu = dxx.tail(nu);

        auto dzx = dz.head(nx);
        auto dzf = dz.tail(nf);
        auto dzs = dzx.head(ns);
        auto dzu = dzx.tail(nu);

        const auto Ax = A.leftCols(nx);
        const auto Af = A.rightCols(nf);
        const auto Au = Ax.rightCols(nu);

        // The variables x arranged in the ordering x = [xs xu xf]
        x.noalias() = state.x(iordering);

        // The variables z arranged in the ordering z = [zs zu zf]
        z.noalias() = state.z(iordering);

        // The gradient of the objective function w.r.t. free variables
        gx.noalias() = f.grad(jx);

        // Calculate ax = -(gx - tr(Ax)*y - zx)
        ax.noalias() = -(gx - tr(Ax) * state.y - zx);

        std::cout << "as = " << tr(as) << std::endl;
        std::cout << "au = " << tr(au) << std::endl;

        // Set af (a for fixed variables) to zero
        af.fill(0.0);

        b.noalias() = -(A*x - params.a);

        // Calculate cx = -(Xx * zx - mu)
        cx.noalias() = -(xx % zx - options.mu);

        // Set cf (c for fixed variables) to zero
        cf.fill(0.0);

        // Calculate as' = as + inv(Xs) * cs
        as += cs/xs;

        std::cout << "as' = " << tr(as) << std::endl;

        // Calculate au' = -(au - Huu * inv(Zu) * cu) // TODO Huu or Huu''? Because here, Huu view actually contains Huu'' values
        au.noalias() = -(au - Huu % cu/zu);

        std::cout << "au' = " << tr(au) << std::endl;

        // Calculate b' = b - Au * inv(Zu) * cu
        b -= Au * (cu/zu);

        // Calculate b'' = b' + Au * Huu'' * au'
        b += Au * (Duu % au);

        dzu = au;
        au.fill(0.0);

        kkt.solve({H, A, G, ns, nu + nf}, {a, b}, {dx, dy});

        dy *= -1.0;

        std::cout << "Duu = " << tr(Duu) << std::endl;
        std::cout << "Duu % (zu/xu) = " << tr(Duu % (zu/xu)) << std::endl;

        dzu = Duu % (zu/xu) % (dzu - tr(Au)*dy);
        dxu = (cu - xu % dzu)/zu;
        dzs = (cs - zs % dxs)/xs;

        iordering.asPermutation().applyThisOnTheLeft(dx);
        iordering.asPermutation().applyThisOnTheLeft(dz);

        VectorXd residual_tmp = residual;
        VectorXd solution_tmp = solution;

        solve2(params, state, f);

        std::cout << "stable  = " << tr(js) << std::endl;
        std::cout << "x       = " << tr(state.x) << std::endl;
        std::cout << "dx(rs)  = " << tr(solution_tmp.head(n)) << std::endl;
        std::cout << "dx(lu)  = " << tr(dx) << std::endl;
        std::cout << "res(dx) = " << tr(abs(solution_tmp.head(n) - dx)) << std::endl;
        std::cout << std::endl;
        std::cout << "y       = " << tr(state.y) << std::endl;
        std::cout << "dy(rs)  = " << tr(solution_tmp.segment(n, m)) << std::endl;
        std::cout << "dy(lu)  = " << tr(dy) << std::endl;
        std::cout << "res(dy) = " << tr(abs(solution_tmp.segment(n, m) - dy)) << std::endl;
        std::cout << std::endl;
        std::cout << "z       = " << tr(state.z) << std::endl;
        std::cout << "dz(rs)  = " << tr(solution_tmp.tail(n)) << std::endl;
        std::cout << "dz(lu)  = " << tr(dz) << std::endl;
        std::cout << "res(dz) = " << tr(abs(solution_tmp.tail(n) - dz)) << std::endl;

        residual = residual_tmp;
        solution = solution_tmp;
    }

    /// Solve the KKT matrix equation.
    auto solve2(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        const auto& x = state.x;
        const auto& y = state.y;
        const auto& z = state.z;
        const auto& A = structure.A;
        const auto& a = params.a;

//        MatrixXd M = zeros(n+m, n+m);
//        if(f.hessian.size())
//            M.topLeftCorner(n, n) = f.hessian;
//        M.topLeftCorner(n, n).diagonal() += z/x;
//        M.topRows(n).rightCols(m) = -tr(A);
//        M.bottomRows(m).leftCols(n) = A;

        MatrixXd M = zeros(t, t);
        if(f.hessian.size())
            M.topLeftCorner(n, n) = f.hessian;
        M.topRows(n).middleCols(n, m) = -tr(A);
        M.topRightCorner(n, n).diagonal().fill(-1.0);
        M.middleRows(n, m).leftCols(n) = A;
        M.bottomLeftCorner(n, n).diagonal() = z;
        M.bottomRightCorner(n, n).diagonal() = x;

        residual.head(n) = -(f.grad - tr(A)*y - z);
        residual.segment(n, m) = -(A*x - a);
        residual.tail(n) = -(x % z - options.mu);

        solution = M.fullPivLu().solve(residual);
    }
};

OptimumStepper::OptimumStepper()
: pimpl(new Impl())
{}

OptimumStepper::OptimumStepper(const OptimumStepper& other)
: pimpl(new Impl(*other.pimpl))
{}

OptimumStepper::~OptimumStepper()
{}

auto OptimumStepper::operator=(OptimumStepper other) -> OptimumStepper&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto OptimumStepper::setOptions(const OptimumOptions& options) -> void
{
    pimpl->options = options;
    pimpl->kkt.setOptions(options.kkt);
}

auto OptimumStepper::initialize(const OptimumStructure& structure) -> void
{
    pimpl->initialize(structure);
}

auto OptimumStepper::decompose(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
{
    pimpl->decompose(params, state, f);
}

auto OptimumStepper::solve(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
{
    pimpl->solve(params, state, f);
}

auto OptimumStepper::solve2(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
{
    pimpl->solve2(params, state, f);
}

auto OptimumStepper::step() const -> VectorXdConstRef
{
    return pimpl->solution;
}

auto OptimumStepper::dx() const -> VectorXdConstRef
{
    return pimpl->solution.head(pimpl->n);
}

auto OptimumStepper::dy() const -> VectorXdConstRef
{
    return pimpl->solution.segment(pimpl->n, pimpl->m);
}

auto OptimumStepper::dz() const -> VectorXdConstRef
{
    return pimpl->solution.tail(pimpl->n);
}

auto OptimumStepper::dw() const -> VectorXdConstRef
{
    return pimpl->solution.tail(pimpl->n);
}

auto OptimumStepper::residual() const -> VectorXdConstRef
{
    return pimpl->residual;
}

auto OptimumStepper::residualOptimality() const -> VectorXdConstRef
{
    return pimpl->residual.head(pimpl->n);
}

auto OptimumStepper::residualFeasibility() const -> VectorXdConstRef
{
    return pimpl->residual.segment(pimpl->n, pimpl->m);
}

auto OptimumStepper::residualComplementarityLowerBounds() const -> VectorXdConstRef
{
    return pimpl->residual.tail(pimpl->n);
}

auto OptimumStepper::residualComplementarityUpperBounds() const -> VectorXdConstRef
{
    return pimpl->residual.tail(pimpl->n);
}

//auto OptimumStepper::residualComplementarityInequality() const -> VectorXdConstRef
//{
//
//}
//
//auto OptimumStepper::lhs() const -> MatrixXdConstRef
//{
//
//}

auto OptimumStepper::ifree() const -> VectorXiConstRef
{
    return pimpl->iordering.head(pimpl->nx);
}

auto OptimumStepper::ifixed() const -> VectorXiConstRef
{
    return pimpl->iordering.tail(pimpl->nf);
}

auto OptimumStepper::istable() const -> VectorXiConstRef
{
    return pimpl->iordering.head(pimpl->ns);
}

auto OptimumStepper::iunstable() const -> VectorXiConstRef
{
    return pimpl->iordering.segment(pimpl->ns, pimpl->nu);
}

} // namespace Optima
