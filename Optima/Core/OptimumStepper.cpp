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

// Eigenx includes
#include <Eigenx/LU.hpp> // todo check if necessary later

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
            "The number of stable variables must be positive.");

        // Permute the columns of A so that A = [As Au Af]
        iordering.asPermutation().applyThisOnTheRight(A);

        // The variables x arranged in the ordering x = [xs xu xf]
        x.noalias() = state.x(iordering);

        // The variables z arranged in the ordering z = [zs zu zf]
        z.noalias() = state.z(iordering);

        // The indices of the variables j = [jx jf], with jx = [js ju]
        const auto jx = iordering.head(nx); // indices of free variables

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsu; Hus Huu]
        auto Hxx = H.topLeftCorner(nx, nx);
        auto Hss = Hxx.topLeftCorner(ns, ns);

        // Views to sub-vectors xs and xu in xx = [xs xu]
        const auto xx = x.head(nx);
        const auto xs = xx.head(ns);

        // Views to sub-vectors zs and zu in zx = [zs zu]
        const auto zx = z.head(nx);
        const auto zs = zx.head(ns);

        // Update Hxx = [Hss Hsu; Hus Huu]
        Hxx.noalias() = f.hessian(jx, jx);

        // Calculate Hss' = Hss + inv(Xs)*Zs
        Hss.diagonal() += zs/xs;

        // Update the decomposition of the KKT matrix
        kkt.reorder(iordering);
        kkt.decompose({H, A, G, ns, nu + nf});
    }

    /// Solve the KKT matrix equation.
    auto solve(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        // The indices of the variables j = [jx jf], with jx = [js ju]
        const auto jx = iordering.head(nx);  // indices of free variables

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsu; Hus Huu]
        auto Hxx = H.topLeftCorner(nx, nx);
        auto Hsu = Hxx.topRightCorner(ns, nu);
        auto Hus = Hxx.bottomLeftCorner(nu, ns);
        auto Huu = Hxx.bottomRightCorner(nu, nu);

        // The gradient of the objective function corresponding to free variables
        auto gx = g.head(nx);

        // The vectors a, b, c
        auto a = residual.head(n);
        auto b = residual.segment(n, m);
        auto c = residual.tail(n);

        // The vector a = [ax af], ax = [as au], and its sub-vectors
        auto ax = a.head(nx);
        auto af = a.tail(nf);
        auto as = ax.head(ns);
        auto au = ax.tail(nu);

        // The vector c = [cx cf], cx = [cs cu], and its sub-vectors
        auto cx = c.head(nx);
        auto cf = c.tail(nf);
        auto cs = cx.head(ns);
        auto cu = cx.tail(nu);

        // The vector x = [xx xf], xx = [xs xu], and its sub-vectors
        auto xx = x.head(nx);
        auto xs = xx.head(ns);
        auto xu = xx.tail(nu);

        // The vector z = [zx zf], zx = [zs zu], and its sub-vectors
        auto zx = z.head(nx);
        auto zs = zx.head(ns);
        auto zu = zx.tail(nu);

        // The vectors dx, dy, dz
        auto dx = solution.head(n);
        auto dy = solution.segment(n, m);
        auto dz = solution.tail(n);

        // The vector dx = [dxx dxf], dxx = [dxs dxu], and its sub-vectors
        auto dxx = dx.head(nx);
        auto dxs = dxx.head(ns);
        auto dxu = dxx.tail(nu);

        // The vector dz = [dzx dzf], dzx = [dzs dzu], and its sub-vectors
        auto dzx = dz.head(nx);
        auto dzs = dzx.head(ns);
        auto dzu = dzx.tail(nu);

        // The matrix A = [As Au Af] and its sub-matrices
        const auto Ax = A.leftCols(nx);
        const auto Au = Ax.rightCols(nu);

        // The variables x arranged in the ordering x = [xs xu xf]
        x.noalias() = state.x(iordering);

        // The variables z arranged in the ordering z = [zs zu zf]
        z.noalias() = state.z(iordering);

        // The gradient of the objective function w.r.t. free variables
        gx.noalias() = f.grad(jx);

        // Calculate ax = -(gx + tr(Ax)*y - zx)
        ax.noalias() = -(gx + tr(Ax) * state.y - zx);

        // Store -au into dzu
        dzu.noalias() = -au;

        // Set both au and af sub-vectors in a = [as au af] to zero
        au.fill(0.0);
        af.fill(0.0);

        // Calculate b = -(A*x - a)
        b.noalias() = -(A*x - params.a);

        // Calculate cx = -(Xx * zx - mu)
        cx.noalias() = -(xx % zx - options.mu);

        // Set cf (c for fixed variables) to zero
        cf.fill(0.0);

        // Calculate as' = as + inv(Xs) * cs
        as += cs/xs - Hsu * (cu/zu);

        // Calculate b' = b - Au * inv(Zu) * cu
        b -= Au * (cu/zu);

        // Solve the reduced KKT equation
        kkt.solve({H, A, G, ns, nu + nf}, {a, b}, {dx, dy});

        // Calculate dzu = -au + Hus*dxs + tr(Au)*dy + Huu*inv(Zu)*cu
        dzu += Hus*dxs + tr(Au)*dy + Huu*(cu/zu);

        // Calculate dxu = inv(Zu) * (cu - Xu * dzu)
        dxu = (cu - xu % dzu)/zu;

        // Calculate dzs = inv(Xs) * (cs - Zs * dxs)
        dzs = (cs - zs % dxs)/xs;

        // Permute the calculated dx and dz to their original order
        iordering.asPermutation().applyThisOnTheLeft(dx);
        iordering.asPermutation().applyThisOnTheLeft(dz);
    }

//    /// Solve the KKT matrix equation.
//    auto solve2(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
//    {
//        const auto& x = state.x;
//        const auto& y = state.y;
//        const auto& z = state.z;
//        const auto& A = structure.A;
//        const auto& a = params.a;
//
//        MatrixXd M = zeros(t, t);
//        if(f.hessian.size())
//            M.topLeftCorner(n, n) = f.hessian;
//        M.topRows(n).middleCols(n, m) = tr(A);
//        M.topRightCorner(n, n).diagonal().fill(-1.0);
//        M.middleRows(n, m).leftCols(n) = A;
//        M.bottomLeftCorner(n, n).diagonal() = z;
//        M.bottomRightCorner(n, n).diagonal() = x;
//
//        residual.head(n) = -(f.grad + tr(A)*y - z);
//        residual.segment(n, m) = -(A*x - a);
//        residual.tail(n) = -(x % z - options.mu);
//
//        solution = M.fullPivLu().solve(residual);
//    }
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
