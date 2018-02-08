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

#include "IpSaddlePointSolver.hpp"


#include <iostream>




// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IpSaddlePointMatrix.hpp>
#include <Optima/SaddlePointMatrix.hpp>
#include <Optima/SaddlePointSolver.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/SaddlePointResult.hpp>
#include <Optima/Canonicalizer.hpp>
#include <Optima/EigenExtern.hpp>
using namespace Eigen;

namespace Optima {

struct IpSaddlePointSolver::Impl
{
    /// The `A` matrix in the KKT equation.
    MatrixXd A;

    /// The `H` matrix in the KKT equation.
    MatrixXd H;

    /// The `G` matrix in the KKT equation.
    MatrixXd G;

    VectorXd Z, W, L, U;

    VectorXd r;

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
    auto initialize(MatrixXdConstRef A) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Initialize the members related to number of variables and constraints
        n  = A.cols();
        m  = A.rows();
        nf = 0;
        nx = n;
        ns = n;
        nu = 0;
        t  = 3*n + m;

        // Initialize the ordering of the variables
        iordering.setLinSpaced(n, 0, n - 1);

        // Allocate memory for some members
        H = zeros(n, n);
        Z = zeros(n);
        W = zeros(n);
        L = zeros(n);
        U = zeros(n);
        r = zeros(t);

        // Initialize the saddle point solver
        res += kkt.initialize(A);

        return res.stop();
    }

    /// Decompose the KKT matrix equation used to compute the step vectors.
    auto decompose(IpSaddlePointMatrix lhs) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

        // Auxiliary variables
        const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

        // Initialize the number of free and fixed variables
        nx = lhs.nx();
        nf = lhs.nf();

        // Initialize auxiliary matrices
        A.noalias() = lhs.A();
        Z.noalias() = lhs.Z();
        W.noalias() = lhs.W();
        L.noalias() = lhs.L();
        U.noalias() = lhs.U();

        // Define the function that determines if variable x[i] is lower unstable
        auto lower_unstable = [&](Index i)
        {
           return std::abs(L[i]) <= eps && std::abs(Z[i]) >= eps && std::abs(W[i]) <= eps;
        };

        // Define the function that determines if variable x[i] is upper unstable
        auto upper_unstable = [&](Index i)
        {
           return std::abs(U[i]) <= eps && std::abs(W[i]) >= eps && std::abs(Z[i]) <= eps;
        };

        // Define the function that determines if variable x[i] is stable
        auto stable = [&](Index i)
        {
           return !lower_unstable(i) && !upper_unstable(i);
        };

        // Partition the free variables into stable and unstable: xx = [x(stable) x(unstable)]
        auto is = std::partition(iordering.data(), iordering.data() + nx, stable);

        // Partition the unstable variables into lower and upper unstable variables: x(unstable) = [xl xu]
        auto il = std::partition(is, iordering.data() + nx, lower_unstable);

        // Update the number of stable, lower unstable, and upper unstable variables
        ns = is - iordering.data();
        nl = il - is;
        nu = nx - ns - nl;

        // Ensure the number of stable variables is positive
        if(ns == 0) return res.failed(
            "Could not decompose the interior-point saddle point matrix, "
            "which is singular and has no stable variables.");

        // Permute A, Z, W, L and U according to iordering
        iordering.asPermutation().transpose().applyThisOnTheLeft(Z);
        iordering.asPermutation().transpose().applyThisOnTheLeft(W);
        iordering.asPermutation().transpose().applyThisOnTheLeft(L);
        iordering.asPermutation().transpose().applyThisOnTheLeft(U);
        iordering.asPermutation().applyThisOnTheRight(A);

        // The indices of the free variables
        const auto jx = iordering.head(nx);

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsl Hsu; Hls Hll Hlu; Hus Hul Huu]
        auto Hxx = H.topLeftCorner(nx, nx);
        auto Hss = H.topLeftCorner(ns, ns);

        // Views to the sub-vectors in Z = [Zs Zl Zu Zf]
        auto Zs = Z.head(ns);
        auto Zf = Z.tail(nf);

        // Views to the sub-vectors in W = [Ws Wl Wu Wf]
        auto Ws = W.head(ns);
        auto Wf = W.tail(nf);

        // Views to the sub-vectors in L = [Ls Ll Lu Lf]
        auto Ls = L.head(ns);
        auto Lf = L.tail(nf);

        // Views to the sub-vectors in U = [Us Ul Uu Uf]
        auto Us = U.head(ns);
        auto Uf = U.tail(nf);

        // Ensure Zf = 0, Wf = 0, Lf = I, and Uf = I
        Zf.fill(0.0);
        Wf.fill(0.0);
        Lf.fill(1.0);
        Uf.fill(1.0);

        // Update Hxx = [Hss Hsl Hsu; Hls Hll Hlu; Hus Hul Huu]
        Hxx.noalias() = lhs.H()(jx, jx);

        // Calculate Hss' = Hss + inv(Ls)*Zs + inv(Us)*Ws
        Hss.diagonal() += Zs/Ls + Ws/Us;

        // Update the ordering of the saddle point solver
        kkt.reorder(iordering);

        // Decompose the saddle point matrix
        res += kkt.decompose({H, A, G, ns, n - ns});

        return res.stop();
    }

    /// Solve the KKT matrix equation.
    auto solve(IpSaddlePointVector rhs, IpSaddlePointSolution sol) -> SaddlePointResult
    {
        // The result of this method call
        SaddlePointResult res;

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

        // Views to the sub-matrices in A = [As Al Au Af]
        const auto Ax = A.leftCols(nx);
        const auto Al = Ax.middleCols(ns, nl);
        const auto Au = Ax.rightCols(nu);

        // Views to the sub-vectors in Z = [Zs Zl Zu Zf]
        const auto Zx = Z.head(nx);
        const auto Zs = Zx.head(ns);
        const auto Zl = Zx.segment(ns, nl);
        const auto Zu = Zx.tail(nu);

        // Views to the sub-vectors in W = [Ws Wl Wu Wf]
        const auto Wx = W.head(nx);
        const auto Ws = Wx.head(ns);
        const auto Wl = Wx.segment(ns, nl);
        const auto Wu = Wx.tail(nu);

        // Views to the sub-vectors in L = [Ls Ll Lu Lf]
        const auto Lx = L.head(nx);
        const auto Ls = Lx.head(ns);
        const auto Ll = Lx.segment(ns, nl);
        const auto Lu = Lx.tail(nu);

        // Views to the sub-vectors in U = [Us Ul Uu Uf]
        const auto Ux = U.head(nx);
        const auto Us = Ux.head(ns);
        const auto Ul = Ux.segment(ns, nl);
        const auto Uu = Ux.tail(nu);

        // The right-hand side vectors [a b c d]
        auto a = r.head(n);
        auto b = r.segment(n, m);
        auto c = r.segment(n + m, n);
        auto d = r.tail(n);

        // The solution vectors [x y z w]
        auto x = sol.x();
        auto y = sol.y();
        auto z = sol.z();
        auto w = sol.w();

        // Views to the sub-vectors in a = [as al au af]
        auto ax = a.head(nx);
        auto as = ax.head(ns);
        auto al = ax.segment(ns, nl);
        auto au = ax.tail(nu);
        auto af = a.tail(nf);

        // Views to the sub-vectors in c = [cs cl cu cf]
        auto cx = c.head(nx);
        auto cs = cx.head(ns);
        auto cl = cx.segment(ns, nl);
        auto cu = cx.tail(nu);
        auto cf = c.tail(nf);

        // Views to the sub-vectors in d = [ds dl du df]
        auto dx = d.head(nx);
        auto ds = dx.head(ns);
        auto dl = dx.segment(ns, nl);
        auto du = dx.tail(nu);
        auto df = d.tail(nf);

        // Views to the sub-vectors in x = [xs xl xu xf]
        auto xx = x.head(nx);
        auto xs = xx.head(ns);
        auto xl = xx.segment(ns, nl);
        auto xu = xx.tail(nu);
        auto xf = x.tail(nf);

        // Views to the sub-vectors in z = [zs zl zu zf]
        auto zx = z.head(nx);
        auto zs = zx.head(ns);
        auto zl = zx.segment(ns, nl);
        auto zu = zx.tail(nu);
        auto zf = z.tail(nf);

        // Views to the sub-vectors in w = [ws wl wu wf]
        auto wx = w.head(nx);
        auto ws = wx.head(ns);
        auto wl = wx.segment(ns, nl);
        auto wu = wx.tail(nu);
        auto wf = w.tail(nf);

        // Initialize a, b, c, d in the ordering x = [xs, xl, xu, xf]
        a.noalias() = rhs.a()(iordering);
        b.noalias() = rhs.b();
        c.noalias() = rhs.c()(iordering);
        d.noalias() = rhs.d()(iordering);

        // Calculate as' = as + inv(Ls)*cs + inv(Us)*ds - Hsl*inv(Zl)*cl - Hsu*inv(Wu)*du
        as += cs/Ls + ds/Us - Hsl*(cl/Zl) - Hsu*(du/Wu);

        zl.noalias() = -al;
        wu.noalias() = -au;

        al.fill(0.0);
        au.fill(0.0);
        af.fill(0.0);

        // Calculate b' = b - Al*inv(Zl)*cl - Au*inv(Wu)*du
        b -= Al*(cl/Zl) + Au*(du/Wu);

        // Solve the saddle point problem
        res += kkt.solve({a, b}, {x, y});

        // Calculate zl and wu
        zl.noalias() += Hls*xs + tr(Al)*y + Hll*(cl/Zl) + Hlu*(du/Wu) - dl/Ul;
        wu.noalias() += Hus*xs + tr(Au)*y + Hul*(cl/Zl) + Huu*(du/Wu) - cu/Lu;

        // Calculate xl and xu
        xl.noalias() = (cl - Ll % zl)/Zl;
        xu.noalias() = (du - Uu % wu)/Wu;
        xf.noalias() = af;

        // Calculate zs and zu
        zs.noalias() = (cs - Zs % xs)/Ls;
        zu.noalias() = (cu - Zu % xu)/Lu;
        zf.noalias() = cf;

        // Calculate ws and wl
        ws.noalias() = (ds - Ws % xs)/Us;
        wl.noalias() = (dl - Wl % xl)/Ul;
        wf.noalias() = df;

        // Permute the calculated (x z w) to their original order
        iordering.asPermutation().applyThisOnTheLeft(x);
        iordering.asPermutation().applyThisOnTheLeft(z);
        iordering.asPermutation().applyThisOnTheLeft(w);

        return res.stop();
    }

    /// Update the order of the variables.
    auto reorder(VectorXiConstRef ordering) -> void
    {
        // Update the ordering of the basic KKT solver
        kkt.reorder(ordering);

        // Update the internal ordering of the variables with the new ordering
        ordering.asPermutation().transpose().applyThisOnTheLeft(iordering);
    }
};

IpSaddlePointSolver::IpSaddlePointSolver()
: pimpl(new Impl())
{}

IpSaddlePointSolver::IpSaddlePointSolver(const IpSaddlePointSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

IpSaddlePointSolver::~IpSaddlePointSolver()
{}

auto IpSaddlePointSolver::operator=(IpSaddlePointSolver other) -> IpSaddlePointSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto IpSaddlePointSolver::setOptions(const SaddlePointOptions& options) -> void
{
    pimpl->kkt.setOptions(options);
}

auto IpSaddlePointSolver::options() const -> const SaddlePointOptions&
{
    return pimpl->kkt.options();
}

auto IpSaddlePointSolver::initialize(MatrixXdConstRef A) -> SaddlePointResult
{
    return pimpl->initialize(A);
}

auto IpSaddlePointSolver::decompose(IpSaddlePointMatrix lhs) -> SaddlePointResult
{
    return pimpl->decompose(lhs);
}

auto IpSaddlePointSolver::solve(IpSaddlePointVector rhs, IpSaddlePointSolution sol) -> SaddlePointResult
{
    return pimpl->solve(rhs, sol);
}

auto IpSaddlePointSolver::reorder(VectorXiConstRef ordering) -> void
{
    pimpl->reorder(ordering);
}

} // namespace Optima
