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
#include <Optima/Common/Exception.hpp>
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

    VectorXd x, z, w, l, u, g; // TODO g is not necessary - storage in residual can be used to store it

    /// The KKT solver.
    SaddlePointSolver kkt;

    /// The order of the variables as `x = [x(stable) x(lower) x(upper) x(fixed)]`.
    VectorXi iordering;

    /// The number of variables.
    Index n;

    /// The current number of stable, lower unstable, upper unstable, free, and fixed variables.
    Index ns, nl, nu, nx, nf;

    /// The number of variables with lower and upper bounds
    Index nlower, nupper;

    /// The number of equality constraints.
    Index m;

    /// The total number of variables (x, y, z, w).
    Index t;

    /// The indices of the fixed variables.
    VectorXi ifixed;

    /// The indices of the variables with and without lower bounds.
    VectorXi ilowerinfo;

    /// The indices of the variables with and without lower bounds.
    VectorXi iupperinfo;

    /// Initialize the stepper with the structure of the optimization problem.
    auto initialize(const OptimumStructure& structure) -> void
    {
        // Initialize the members related to number of variables and constraints
        n  = structure.n;
        m  = structure.A.rows();
        nlower = structure.ilower().size();
        nupper = structure.iupper().size();
        nf = structure.ifixed().size();
        nx = n - nf;
        ns = n;
        nu = 0;
        t  = 3*n + m;

        // Initialize the indices of the fixed variables and those with lower/upper bounds
        ifixed = structure.ifixed();

        // Initialize the ordering of the variables
        iordering.setLinSpaced(n, 0, n - 1);
        iordering.tail(nf).swap(iordering(ifixed));

        // Initialize the indices of the variables with/without lower/upper bounds
        ilowerinfo.setLinSpaced(n, 0, n - 1);
        iupperinfo.setLinSpaced(n, 0, n - 1);
        ilowerinfo.head(nlower).swap(ilowerinfo(structure.ilower()));
        iupperinfo.head(nupper).swap(iupperinfo(structure.iupper()));

        // Allocate memory for some members
        A = structure.A;
        H = zeros(n, n);
        g = zeros(n);
        x = zeros(n);
        z = zeros(n);
        w = zeros(n);
        l = zeros(n);
        u = zeros(n);
        residual = zeros(t);
        solution = zeros(t);

        // Initialize the saddle point solver
        kkt.initialize(structure.A);
    }

    /// Decompose the KKT matrix equation used to compute the step vectors.
    auto decompose(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        // Permute the columns of A back to its original order before they are reordered again
        iordering.asPermutation().transpose().applyThisOnTheRight(A);

        // Alias to OptimumParams members
        const auto xlower = params.xlower();
        const auto xupper = params.xupper();

        // The indices of the variables with lower and upper bounds
        const auto ilower = ilowerinfo.head(nlower);
        const auto iupper = iupperinfo.head(nupper);

        // The indices of the variables without lower and upper bounds
        const auto inolower = ilowerinfo.tail(n - nlower);
        const auto inoupper = iupperinfo.tail(n - nupper);

        // Auxiliary variables
        const double mu = options.mu;
        const double eps = std::sqrt(mu);

        // Initialize the values of z(bar) and w(bar)
        z(inolower).fill(0.0);
        w(inoupper).fill(0.0);
        z(ilower) = state.z(ilower);
        w(iupper) = state.w(iupper);

        // Initialize the values of l(bar) and u(bar)
        l(inolower).fill(1.0);
        u(inoupper).fill(1.0);
        l(ilower) = state.x(ilower) - params.xlower();
        u(iupper) = state.x(iupper) - params.xupper();

        // Define the function that determines if variable x[i] is lower unstable
        auto lower_unstable = [&](Index i)
        {
           return std::abs(l[i]) <= eps && std::abs(z[i]) >= eps;
        };

        // Define the function that determines if variable x[i] is upper unstable
        auto upper_unstable = [&](Index i)
        {
           return std::abs(u[i]) <= eps && std::abs(w[i]) >= eps;
        };

        // Define the function that determines if variable x[i] is stable
        auto stable = [&](Index i)
        {
           return !lower_unstable(i) && !upper_unstable(i);
        };

        // Partition the free variables into stable and unstable: xx = [x(stable) x(unstable)]
        auto is = std::partition(iordering.data(), iordering.data() + nx, stable);

        // Partition the stable and lower unstable variables: xsl = [xs xl]
        auto il = std::partition(is, iordering.data() + nx, lower_unstable);

        // Update the number of stable, lower unstable, and upper unstable variables
        ns = is - iordering.data();
        nl = il - is;
        nu = nx - ns - nl;

        // Ensure the number of stable variables is positive
        Assert(ns > 0, "Could not compute the step.",
           "The number of stable variables must be positive.");

        // The variables x arranged in the ordering x = [xs xl xu xf]
        x.noalias() = state.x(iordering);

        // Permute z(bar), w(bar), l(bar) and u(bar) according to iordering
        iordering.asPermutation().transpose().applyThisOnTheLeft(z);
        iordering.asPermutation().transpose().applyThisOnTheLeft(w);
        iordering.asPermutation().transpose().applyThisOnTheLeft(l);
        iordering.asPermutation().transpose().applyThisOnTheLeft(u);

        // Permute the columns of A so that A = [As Al Au Af]
        iordering.asPermutation().applyThisOnTheRight(A);

        // The indices of the free variables
        const auto jx = iordering.head(nx);

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsl Hsu; Hls Hll Hlu; Hus Hul Huu]
        auto Hxx = H.topLeftCorner(nx, nx);
        auto Hss = H.topLeftCorner(ns, ns);

        // Views to the sub-vectors in z = [zs zl zu zf]
        auto zs = z.head(ns);

        // Views to the sub-vectors in w = [ws wl wu wf]
        auto ws = w.head(ns);

        // Views to the sub-vectors in l = [ls ll lu lf]
        auto ls = l.head(ns);

        // Views to the sub-vectors in u = [us ul uu uf]
        auto us = u.head(ns);

        // Update Hxx = [Hss Hsu; Hus Huu]
        Hxx.noalias() = f.hessian(jx, jx);

        // Calculate Hss' = Hss + inv(Ls)*Zs + inv(Us)*Ws
        Hss.diagonal() += zs/ls + ws/us;

        // Update the ordering of the saddle point solver
        kkt.reorder(iordering);

        // Decompose the saddle point matrix
        kkt.decompose({H, A, G, ns, n - ns});
    }

    /// Solve the KKT matrix equation.
    auto solve(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        // Auxiliary variables
        const double mu = options.mu;

        // The indices of the free variables
        const auto jx = iordering.head(nx);

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsl Hsu; Hls Hll Hlu; Hus Hul Huu]
        const auto Hxx = H.topLeftCorner(nx, nx);
        const auto Hs  = Hxx.topRows(ns);
        const auto Hl  = Hxx.middleRows(ns, nl);
        const auto Hu  = Hxx.bottomRows(nu);
        const auto Hsl = Hs.middleCols(ns, nl);
        const auto Hsu = Hs.rightCols(nu);
        const auto Hls = Hl.leftCols(ns);
        const auto Hll = Hl.middleCols(ns, nl);
        const auto Hlu = Hl.rightCols(nu);
        const auto Hus = Hu.leftCols(ns);
        const auto Hul = Hu.middleCols(ns, nl);
        const auto Huu = Hu.rightCols(nu);

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
        auto as = ax.head(ns);
        auto al = ax.segment(ns, nl);
        auto au = ax.tail(nu);

        // Views to the sub-vectors in c = [cs cl cu cf]
        auto cx = c.head(nx);
        auto cf = c.tail(nf);
        auto cs = cx.head(ns);
        auto cl = cx.segment(ns, nl);
        auto cu = cx.tail(nu);

        // Views to the sub-vectors in d = [ds dl du df]
        auto dx = d.head(nx);
        auto df = d.tail(nf);
        auto ds = dx.head(ns);
        auto dl = dx.segment(ns, nl);
        auto du = dx.tail(nu);

        // Views to the sub-vectors in z = [zs zl zu zf]
        auto zx = z.head(nx);
        auto zs = zx.head(ns);
        auto zl = zx.segment(ns, nl);
        auto zu = zx.tail(nu);

        // Views to the sub-vectors in w = [ws wl wu wf]
        auto wx = w.head(nx);
        auto ws = wx.head(ns);
        auto wl = wx.segment(ns, nl);
        auto wu = wx.tail(nu);

        // Views to the sub-vectors in l = [ls ll lu lf]
        auto lx = l.head(nx);
        auto ls = lx.head(ns);
        auto ll = lx.segment(ns, nl);
        auto lu = lx.tail(nu);

        // Views to the sub-vectors in u = [us ul uu uf]
        auto ux = u.head(nx);
        auto us = ux.head(ns);
        auto ul = ux.segment(ns, nl);
        auto uu = ux.tail(nu);

        // Views to the sub-vectors in the vector solution = [delta(x) delta(y) delta(z) delta(w)]
        auto hx = solution.head(n);
        auto hy = solution.segment(n, m);
        auto hz = solution.segment(n + m, n);
        auto hw = solution.tail(n);

        // Views to the sub-vectors in delta(x)
        auto hxx = hx.head(nx);
        auto hxs = hxx.head(ns);
        auto hxl = hxx.segment(ns, nl);
        auto hxu = hxx.tail(nu);

        // Views to the sub-vectors in delta(z)
        auto hzx = hz.head(nx);
        auto hzs = hzx.head(ns);
        auto hzl = hzx.segment(ns, nl);
        auto hzu = hzx.tail(nu);

        // Views to the sub-vectors in delta(w)
        auto hwx = hw.head(nx);
        auto hws = hwx.head(ns);
        auto hwl = hwx.segment(ns, nl);
        auto hwu = hwx.tail(nu);

        // Views to the sub-matrices in A = [As Al Au Af]
        const auto Ax = A.leftCols(nx);
        const auto Al = Ax.middleCols(ns, nl);
        const auto Au = Ax.rightCols(nu);

        // Initialize the gradient of the objective function w.r.t. free and fixed variables
        gx.noalias() = f.grad(jx);
        gf.fill(0.0);

        // Calculate ax = -(gx + tr(Ax)*y - zx - wx)
        ax.noalias() = -(gx + tr(Ax) * state.y - zx - wx);

        // Store -al into dzl and -au into dwu
        hzl.noalias() = -al;
        hwu.noalias() = -au;

        // Set sub-vectors (al, au, af) in a to zero
        al.fill(0.0);
        au.fill(0.0);
        af.fill(0.0);

        // Calculate b = -(A*x - b)
        b.noalias() = -(A*x - params.b());

        // Calculate both c and d vectors
        cx.noalias() = mu - l % z;
        dx.noalias() = mu - u % w;
        cf.fill(0.0);
        df.fill(0.0);

        // Calculate as' = as + inv(Ls)*cs + inv(Us)*ds - Hsl*inv(Zl)*cl - Hsu*inv(Wu)*du
        as += cs/ls + ds/us - Hsl*(cl/zl) - Hsu*(du/wu);

        // Calculate b' = b - Al*inv(Zl)*cl - Au*inv(Wu)*du
        b -= Al*(cl/zl) + Au*(du/wu);

        // Solve the saddle point problem
        kkt.solve({a, b}, {hx, hy});

        // Calculate dzl and dwu
        hzl += Hls*hxs + tr(Al)*hy + Hll*(cl/zl) + Hlu*(du/wu) - dl/ul;
        hwu += Hus*hxs + tr(Au)*hy + Hul*(cl/zl) + Huu*(du/wu) - cu/lu;

        // Calculate dxl and dxu
        hxl = (cl - ll % hzl)/zl;
        hxu = (du - uu % hwu)/wu;

        // Calculate dzs and dzu
        hzs = (cs - zs % hxs)/ls;
        hzu = (cu - zu % hxu)/lu;

        // Calculate dws and dwl
        hws = (ds - ws % hxs)/us;
        hwl = (dl - wl % hxl)/ul;

        // Permute the calculated (dx dz dw) to their original order
        iordering.asPermutation().applyThisOnTheLeft(hx);
        iordering.asPermutation().applyThisOnTheLeft(hz);
        iordering.asPermutation().applyThisOnTheLeft(hw);
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
