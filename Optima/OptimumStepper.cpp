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


#include <iostream>

// Eigen includes
#include <eigen3/Eigen/LU> // todo check if necessary later

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IpSaddlePointMatrix.hpp>
#include <Optima/IpSaddlePointSolver.hpp>
#include <Optima/OptimumOptions.hpp>
#include <Optima/OptimumParams.hpp>
#include <Optima/OptimumState.hpp>
#include <Optima/OptimumStructure.hpp>
#include <Optima/SaddlePointResult.hpp>
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

    VectorXd Z, W, L, U;

    VectorXd x, g;

    /// The KKT solver.
    IpSaddlePointSolver kkt;

    /// The order of the variables as `x = [x(stable) x(lower) x(upper) x(fixed)]`.
    VectorXi iordering;

    /// The number of variables.
    Index n;

    /// The current number of free and fixedvariables.
    Index nx, nf;

    /// The number of equality constraints.
    Index m;

    /// The total number of variables (x, y, z, w).
    Index t;

    Impl(const OptimumStructure& structure)
    : structure(structure)
    {
        // Initialize the members related to number of variables and constraints
        n  = structure.n;
        m  = structure.A.rows();
        nf = structure.iwithfixed().size();
        nx = n - nf;
        t  = 3*n + m;

        // The ordering of the variables partitioned as [free variables, fixed variables]
        iordering = structure.fixedpartition();

        // Allocate memory for some members
        A.noalias() = structure.A * iordering.asPermutation();
        H = zeros(n, n);
        g = zeros(n);
        x = zeros(n);
        Z = zeros(n);
        W = zeros(n);
        L = zeros(n);
        U = zeros(n);
        residual = zeros(t);
        solution = zeros(t);

        // Initialize the saddle point solver
        kkt.initialize(A);
    }

    /// Decompose the KKT matrix equation used to compute the step vectors.
    auto decompose(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        // The lower and upper bounds of x
        const auto xlower = params.xlower();
        const auto xupper = params.xupper();

        // The indices of the variables with/without lower/upper bounds
        const auto iwithlower = structure.iwithlower();
        const auto iwithupper = structure.iwithupper();
        const auto iwithoutlower = structure.iwithoutlower();
        const auto iwithoutupper = structure.iwithoutupper();

        // Initialize the diagonal matrices Z and W
        Z(iwithlower) = state.z(iwithlower);
        Z(iwithoutlower).fill(0.0);
        W(iwithupper) = state.w(iwithupper);
        W(iwithoutlower).fill(0.0);

        // Initialize the diagonal matrices L and U
        L(iwithlower) = state.x(iwithlower) - xlower;
        L(iwithoutlower).fill(1.0);
        U(iwithupper) = state.x(iwithupper) - xupper;
        U(iwithoutlower).fill(1.0);

        // The variables x arranged in the ordering x = [xs xl xu xf]
        x.noalias() = state.x(iordering);

        // Permute the columns of A, Z, W, L and U according to ordering x = [xx xf]
        iordering.asPermutation().transpose().applyThisOnTheLeft(Z); // TODO maybe a swap of columsn would be more efficient
        iordering.asPermutation().transpose().applyThisOnTheLeft(W);
        iordering.asPermutation().transpose().applyThisOnTheLeft(L);
        iordering.asPermutation().transpose().applyThisOnTheLeft(U);

        // The indices of the free variables
        const auto jx = iordering.head(nx);

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsl Hsu; Hls Hll Hlu; Hus Hul Huu]
        auto Hxx = H.topLeftCorner(nx, nx);

        // Update Hxx = [Hss Hsu; Hus Huu]
        Hxx.noalias() = f.hessian(jx, jx);

        std::cout << "matrix = \n" << IpSaddlePointMatrix(H, A, Z, W, L, U, nx, nf).matrix() << std::endl;

        // Decompose the interior-point saddle point matrix
        kkt.decompose({H, A, Z, W, L, U, nx, nf});
    }

    /// Solve the KKT matrix equation.
    auto solve(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        // Auxiliary variables
        const double mu = options.mu;

        // The indices of the free variables
        const auto jx = iordering.head(nx);

        // The gradient of the objective function corresponding to free variables
        auto gx = g.head(nx);
        auto gf = g.tail(nf);

        // Views to the sub-vectors in residual = [a b c d]
        auto a = residual.head(n);
        auto b = residual.segment(n, m);
        auto c = residual.segment(n + m, n);
        auto d = residual.tail(n);

        // Views to the sub-vectors in a = [as al au af]
        auto ax = a.head(nx);
        auto af = a.tail(nf);

        // Views to the sub-vectors in c = [cs cl cu cf]
        auto cx = c.head(nx);
        auto cf = c.tail(nf);

        // Views to the sub-vectors in d = [ds dl du df]
        auto dx = d.head(nx);
        auto df = d.tail(nf);

        // Views to the sub-vectors in z = [zs zl zu zf]
        auto zx = Z.head(nx);

        // Views to the sub-vectors in w = [ws wl wu wf]
        auto wx = W.head(nx);

        // Views to the sub-vectors in the vector solution = [delta(x) delta(y) delta(z) delta(w)]
        auto hx = solution.head(n);
        auto hy = solution.segment(n, m);
        auto hz = solution.segment(n + m, n);
        auto hw = solution.tail(n);

        // Views to the sub-matrices in A = [As Al Au Af]
        const auto Ax = A.leftCols(nx);

        // Initialize the gradient of the objective function w.r.t. free and fixed variables
        gx.noalias() = f.grad(jx);
        gf.fill(0.0);

        // Calculate ax = -(gx + tr(Ax)*y - zx - wx)
        ax.noalias() = -(gx + tr(Ax) * state.y - zx - wx);
        af.fill(0.0);

//        // Set sub-vectors (al, au, af) in a to zero
//        al.fill(0.0);
//        au.fill(0.0);
//        af.fill(0.0);

        // Calculate b = -(A*x - b)
        b.noalias() = -(A*x - params.b());

        // Calculate both c and d vectors
        cx.noalias() = mu - L % Z; // TODO Try mu - L % Z and -L % Z
        dx.noalias() = mu - U % W; // TODO Try mu - U % W and -U % W
        cf.fill(0.0);
        df.fill(0.0);

        std::cout << "residual = " << tr(residual) << std::endl;

        // Solve the saddle point problem
        kkt.solve({a, b, c, d}, {hx, hy, hz, hw});

        // Permute the calculated (dx dz dw) to their original order
        iordering.asPermutation().applyThisOnTheLeft(hx);
        iordering.asPermutation().applyThisOnTheLeft(hz);
        iordering.asPermutation().applyThisOnTheLeft(hw);
    }
};

OptimumStepper::OptimumStepper(const OptimumStructure& structure)
: pimpl(new Impl(structure))
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
//
//auto OptimumStepper::ifree() const -> VectorXiConstRef
//{
//    return pimpl->iordering.head(pimpl->nx);
//}
//
//auto OptimumStepper::ifixed() const -> VectorXiConstRef
//{
//    return pimpl->iordering.tail(pimpl->nf);
//}
//
//auto OptimumStepper::istable() const -> VectorXiConstRef
//{
//    return pimpl->iordering.head(pimpl->ns);
//}
//
//auto OptimumStepper::iunstable() const -> VectorXiConstRef
//{
//    return pimpl->iordering.segment(pimpl->ns, pimpl->nu);
//}

} // namespace Optima
