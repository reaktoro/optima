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

    /// The number of equality constraints.
    Index m;

    /// The total number of variables (x, y, z, w).
    Index t;

    /// Initialize the stepper with the structure of the optimization problem.
    auto initialize(const OptimumStructure& structure) -> void
    {
        // Initialize the members related to number of variables and constraints
        n  = structure.n;
        ns = n;
        nu = 0;
        nx = n;
        nf = 0;
        m  = structure.A.rows();
        t  = 3*n + m;

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
        // Alias to OptimumParams members
        const auto ifixed = params.ifixed();
        const auto xlower = params.xlower();
        const auto xupper = params.xupper();

        // Auxiliary variables
        const double eps = std::sqrt(options.mu);

        // Update the number of fixed and free variables
        nf = ifixed.size();
        nx = n - nf;

        // Partition the variables into free and fixed variables x = [xx xf]
        iordering.tail(nf).swap(iordering(ifixed));

        // Define the function that determines if variable x[i] is lower unstable
        auto lower_unstable = [&](Index i)
        {
            return state.x[i] <= xlower[i] + eps && state.z[i] >= eps;
        };

        // Define the function that determines if variable x[i] is upper unstable
        auto upper_unstable = [&](Index i)
        {
            return state.x[i] >= xupper[i] - eps && state.w[i] <= -eps;
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
        assert(ns > 0 && "Could not compute the step."
            "The number of stable variables must be positive.");

        // Permute the columns of A so that A = [As Al Au Af]
        iordering.asPermutation().applyThisOnTheRight(A);

        // The variables x arranged in the ordering x = [xs xl xu xf]
        x.noalias() = state.x(iordering);

        // The variables z arranged in the ordering z = [zs xl zu zf]
        z.noalias() = state.z(iordering);

        // The variables w arranged in the ordering w = [ws wl wu wf]
        w.noalias() = state.w(iordering);

        // The lower bounds for x arranged in the ordering l = [ls ll lu lf]
        l.noalias() = xlower(iordering);

        // The upper bounds for x arranged in the ordering u = [us ul uu uf]
        u.noalias() = xupper(iordering);

        // Calculate l' = x - l and u' = x - u
        l.noalias() = x - l;
        u.noalias() = x - u;

        // The indices of the free variables
        const auto jx = iordering.head(nx);

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsu; Hus Huu]
        auto Hxx = H.topLeftCorner(nx, nx);
        auto Hss = Hxx.topLeftCorner(ns, ns);

        // Views to stable sub-vectors zs and ws in vectors z and w
        const auto zs = z.head(ns);
        const auto ws = w.head(ns);

        // Views to lower unstable sub-vector zl and upper unstable wu
        auto zl = z.segment(ns, nl);
        auto wu = w.segment(ns + nl, nu);

        // Views to stable and upper unstable sub-vectors ls and lu in vector l
        auto ls = l.head(ns);
        auto lu = l.segment(ns + nl, nu);

        // Views to stable and lower unstable sub-vectors us and ul in vector u
        auto us = u.head(ns);
        auto ul = u.segment(ns, nl);

        // Calculate ls'' = inv(ls') and us'' = inv(us')
        ls.noalias() = 1.0/ls;
        us.noalias() = 1.0/us;

        // Calculate lu'' = inv(lu') and ul'' = inv(uu')
        lu.noalias() = 1.0/lu;
        ul.noalias() = 1.0/ul;

        // Calculate zl' = inv(zl) and wu' = inv(wu)
        zl.noalias() = 1.0/zl;
        wu.noalias() = 1.0/wu;

        // Update Hxx = [Hss Hsu; Hus Huu]
        Hxx.noalias() = f.hessian(jx, jx);

        // Calculate Hss' = Hss + Ls'' * Zs + Us'' * Ws
        Hss.diagonal() += ls % zs + us % ws;

        // Update the decomposition of the KKT matrix
        kkt.reorder(iordering);
        kkt.decompose({H, A, G, ns, n - ns});
    }

    /// Solve the KKT matrix equation.
    auto solve(const OptimumParams& params, const OptimumState& state, const ObjectiveState& f) -> void
    {
        // The indices of the free variables
        const auto jx = iordering.head(nx);

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsl Hsu; Hls Hll Hlu; Hus Hul Huu]
        const auto Hxx = H.topLeftCorner(nx, nx);
        const auto Hs = Hxx.topRows(ns);
        const auto Hl = Hxx.middleRows(ns, nl);
        const auto Hu = Hxx.bottomRows(nu);
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

        // The variables x arranged in the ordering x = [xs xl xu xf]
        x.noalias() = state.x(iordering);

        // The variables z arranged in the ordering z = [zs xl zu zf]
        z.noalias() = state.z(iordering);

        // The variables w arranged in the ordering w = [ws wl wu wf]
        w.noalias() = state.w(iordering);

        // The gradient of the objective function w.r.t. free variables
        gx.noalias() = f.grad(jx);

        // Calculate ax = -(gx + tr(Ax)*y - zx - wx)
        ax.noalias() = -(gx + tr(Ax) * state.y - zx - wx);

        // Store -al into dzl and -au into dwu
        hzl.noalias() = -al;
        hwu.noalias() = -au;

        // Set sub-vectors (al, au, af) in a to zero
        al.fill(0.0);
        au.fill(0.0);
        af.fill(0.0);

        // Calculate b = -(A*x - a)
        b.noalias() = -(A*x - params.b());

        // Calculate cx = -(Zx * (xx - lx) - mu)
        cx.noalias() = -(zx/lx - options.mu);

        // Calculate dx = -(Wx * (xx - ux) - mu)
        dx.noalias() = -(wx/ux - options.mu);

        // Set cf and df to zero
        cf.fill(0.0);
        df.fill(0.0);

        // Calculate as' = as + Ls''*cs + Us''*ds - Hsl*Zl'*cl - Hsu*Wu'*du
        as += ls % cs + us % ds - Hsl*(zl % cl) - Hsu*(wu % du);

        // Calculate b' = b - Al*Zl'cl - Au*Wu'*du
        b -= Al*(zl % cl) + Au*(wu % du);

        // Solve the reduced KKT equation
        kkt.solve({a, b}, {hx, hy});

        // Calculate dzl and dwu
        hzl += Hls*hxs + tr(Al)*hy + Hll*(zl % cl) + Hlu*(wu % du) - ul % dl;
        hwu += Hus*hxs + tr(Au)*hy + Hul*(zl % cl) + Huu*(wu % du) - lu % cu;

        // Calculate dxl and dxu
        hxl = (cl - ll % hzl) % zl;
        hxu = (du - uu % hwu) % wu;

        // Calculate dzs and dzu
        hzs = (cs - zs % hxs) % ls;
        hzu = (cu - zu % hxu) % lu;

        // Calculate dws and dwl
        hws = (ds - ws % hxs) % us;
        hwl = (dl - wl % hxl) % ul;

        // Permute the calculated (dx dz dw) to their original order
        iordering.asPermutation().applyThisOnTheLeft(hx);
        iordering.asPermutation().applyThisOnTheLeft(hz);
        iordering.asPermutation().applyThisOnTheLeft(hw);
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
