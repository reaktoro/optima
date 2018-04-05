// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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

#include "IpSaddlePointSolver.hpp"

// Optima includes
#include <Optima/Canonicalizer.hpp>
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/IpSaddlePointMatrix.hpp>
#include <Optima/Result.hpp>
#include <Optima/SaddlePointMatrix.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/SaddlePointSolver.hpp>
#include <Optima/Utils.hpp>
#include <Optima/VariantMatrix.hpp>
using namespace Eigen;
using namespace Eigen::placeholders;

namespace Optima {

struct IpSaddlePointSolver::Impl
{
    /// The `A` matrix in the saddle point equation.
    Matrix A;

    /// The `H` matrix in the saddle point equation.
    VariantMatrix H;

    /// The matrices D, Z, W, L, U
    Vector D, Z, W, L, U;

    /// The residual vector
    Vector r;

    /// The solution vector
    Vector s;

    /// The saddle point solver.
    SaddlePointSolver spsolver;

    /// The order of the variables as x = [xs xl xu xz xw xf].
    Indices iordering;

    /// The number of variables.
    Index n;

    /// The number of free and fixed variables.
    Index nx, nf;

    /// The number of variables in the partitions (s, l, u, z, w)
    Index ns, nl, nu, nz, nw;

    /// The number of equality constraints.
    Index m;

    /// The total number of variables (x, y, z, w).
    Index t;

    /// Initialize the stepper with the structure of the optimization problem.
    auto initialize(MatrixConstRef A) -> Result
    {
        // The result of this method call
        Result res;

        // Create a copy of matrix A
        this->A = A;

        // Initialize the members related to number of variables and constraints
        n  = A.cols();
        m  = A.rows();
        nx = n;
        nf = 0;
        ns = n;
        nl = 0;
        nu = 0;
        nz = 0;
        nw = 0;
        t  = 3*n + m;

        // Initialize the ordering of the variables
        iordering = indices(n);

        // Allocate memory for some vector/matrix members
        Z = zeros(n);
        W = zeros(n);
        L = zeros(n);
        U = zeros(n);
        r = zeros(t);
        s = zeros(t);

        // Initialize the saddle point solver
        res += spsolver.initialize(A);

        return res.stop();
    }

    /// Update the partitioning of the free variables into (s, l, u, z, w)
    auto updatePartitioning(IpSaddlePointMatrix lhs) -> Result
    {
        // The result of this method call
        Result res;

        // Initialize the number of free and fixed variables
        nf = lhs.jf.size();
        nx = n - nf;

        // Partition the variables into (free variables, fixed variables)
        partitionRight(iordering, lhs.jf);

        // Auxiliary variables
        const double eps = std::numeric_limits<double>::epsilon();

        // Define the functions to classify the variables into the partitions (s, l, u, z, w)
        auto partition_l = [&](Index i) { return std::abs(lhs.L[i]) < std::abs(lhs.Z[i]) && std::abs(lhs.L[i]/lhs.Z[i])  > eps; };
        auto partition_u = [&](Index i) { return std::abs(lhs.U[i]) < std::abs(lhs.W[i]) && std::abs(lhs.U[i]/lhs.W[i])  > eps; };
        auto partition_z = [&](Index i) { return std::abs(lhs.L[i]) < std::abs(lhs.Z[i]) && std::abs(lhs.L[i]/lhs.Z[i]) <= eps; };
        auto partition_w = [&](Index i) { return std::abs(lhs.U[i]) < std::abs(lhs.W[i]) && std::abs(lhs.U[i]/lhs.W[i]) <= eps; };

        // The begin and end iterators in iordering for the free variables
        auto ib = iordering.data();
        auto ie = iordering.data() + nx;

        // Order the free variables into the partitions (w, z, u, l, s)
        auto iw = std::partition(ib, ie, partition_w);
        auto iz = std::partition(iw, ie, partition_z);
        auto iu = std::partition(iz, ie, partition_u);
        auto il = std::partition(iu, ie, partition_l);
        auto is = ie;

        // Reverse the order of the free variables into (s, l, u, z, w)
        std::reverse(ib, ie);

        // Update the number of (s, l, u, z, w) variables
        nw = iw - ib;
        nz = iz - iw;
        nu = iu - iz;
        nl = il - iu;
        ns = is - il;

        // Initialize A, Z, W, L and U according to iordering
        Z.noalias() = lhs.Z;
        W.noalias() = lhs.W;
        L.noalias() = lhs.L;
        U.noalias() = lhs.U;

        // Ensure Z(fixed) = 0, W(fixed) = 0, L(fixed) = 1, and U(fixed) = 1
        Z(lhs.jf).fill(0.0);
        W(lhs.jf).fill(0.0);
        L(lhs.jf).fill(1.0);
        U(lhs.jf).fill(1.0);

        return res.stop();
    }

    /// Decompose the saddle point matrix equation.
    auto decompose(IpSaddlePointMatrix lhs) -> Result
    {
        switch(lhs.H.structure) {
        case MatrixStructure::Dense: return decomposeDenseHessianMatrix(lhs);
        case MatrixStructure::Diagonal: return decomposeDiagonalHessianMatrix(lhs);
        case MatrixStructure::Zero: return decomposeDiagonalHessianMatrix(lhs);
        }

        return {};
    }

    /// Decompose the saddle point matrix equation with dense Hessian matrix.
    auto decomposeDenseHessianMatrix(IpSaddlePointMatrix lhs) -> Result
    {
        // The result of this method call
        Result res;

        // Update the partitioning of the variables
        res += updatePartitioning(lhs);

        // The indices of the free and fixed variables
        const auto jx = iordering.head(ns + nl + nu);
        const auto jf = iordering.tail(nz + nw + nf);

        // Set the Hessian matrix to dense structure
        H.setDense(n);

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsl Hsu; Hls Hll Hlu; Hus Hul Huu]
        H.dense.topLeftCorner(nx, nx) = lhs.H.dense(jx, jx);

        // Calculate D = inv(L)*Z + inv(U)*W
        D += Z/L + W/U;

        // Define the saddle point matrix
        SaddlePointMatrix spm(lhs.H, D, A, jf);

        // Decompose the saddle point matrix
        res += spsolver.decompose(spm);

        return res.stop();
    }

    /// Decompose the saddle point matrix equation with diagonal Hessian matrix.
    auto decomposeDiagonalHessianMatrix(IpSaddlePointMatrix lhs) -> Result
    {
        // The result of this method call
        Result res;

        // Update the partitioning of the variables
        res += updatePartitioning(lhs);

        // The indices of the free and fixed variables
        const auto jx = iordering.head(ns + nl + nu);
        const auto jf = iordering.tail(nz + nw + nf);

        // Views to the sub-vectors in (Z, W, L, U) corresponding to (s, l, u) partition
        auto Ze = Z(jx);
        auto We = W(jx);
        auto Le = L(jx);
        auto Ue = U(jx);

        // Set the Hessian matrix to diagonal structure
        H.setDiagonal(n);

        // Views to the blocks of the Hessian matrix Hxx = [Hss Hsl Hsu; Hls Hll Hlu; Hus Hul Huu]
        auto Hee = H.diagonal(jx);

        // Update Hxx
        Hee = lhs.H.diagonal(jx);

        // Calculate Hee' = Hee + inv(Le)*Ze + inv(Ue)*We
        Hee += Ze/Le + We/Ue;

        // Define the saddle point matrix
        SaddlePointMatrix spm(lhs.H, D, A, jf);

        // Decompose the saddle point matrix
        res += spsolver.decompose(spm);

        // Remove the previously added vector along the diagonal of the Hessian
        Hee -= Ze/Le + We/Ue;

        return res.stop();
    }

    /// Solve the saddle point matrix equation.
    auto solve(IpSaddlePointVector rhs, IpSaddlePointSolution sol) -> Result
    {
        switch(H.structure) {
        case MatrixStructure::Dense: return solveDenseHessianMatrix(rhs, sol);
        case MatrixStructure::Diagonal: return solveDiagonalHessianMatrix(rhs, sol);
        case MatrixStructure::Zero: return solveDiagonalHessianMatrix(rhs, sol);
        }

        return {};
    }

    /// Solve the saddle point matrix equation with dense Hessian.
    auto solveDenseHessianMatrix(IpSaddlePointVector rhs, IpSaddlePointSolution sol) -> Result
    {
        // The result of this method call
        Result res;

        // Views to the blocks of the Hessian matrix Hxx
        const auto Hxx = H.dense.topLeftCorner(nx, nx);

        const auto Hs  = Hxx.topRows(ns);
        const auto Hl  = Hxx.middleRows(ns, nl);
        const auto Hu  = Hxx.middleRows(ns + nl, nu);
        const auto Hz  = Hxx.middleRows(ns + nl + nu, nz);
        const auto Hw  = Hxx.bottomRows(nw);

        const auto Hsl = Hs.middleCols(ns, nl);
        const auto Hsu = Hs.middleCols(ns + nl, nu);
        const auto Hsz = Hs.middleCols(ns + nl + nu, nz);
        const auto Hsw = Hs.rightCols(nw);

        const auto Hll = Hl.middleCols(ns, nl);
        const auto Hlu = Hl.middleCols(ns + nl, nu);
        const auto Hlz = Hl.middleCols(ns + nl + nu, nz);
        const auto Hlw = Hl.rightCols(nw);

        const auto Hul = Hu.middleCols(ns, nl);
        const auto Huu = Hu.middleCols(ns + nl, nu);
        const auto Huz = Hu.middleCols(ns + nl + nu, nz);
        const auto Huw = Hu.rightCols(nw);

        const auto Hzs = Hz.leftCols(ns);
        const auto Hzl = Hz.middleCols(ns, nl);
        const auto Hzu = Hz.middleCols(ns + nl, nu);
        const auto Hzz = Hz.middleCols(ns + nl + nu, nz);
        const auto Hzw = Hz.rightCols(nw);

        const auto Hws = Hw.leftCols(ns);
        const auto Hwl = Hw.middleCols(ns, nl);
        const auto Hwu = Hw.middleCols(ns + nl, nu);
        const auto Hwz = Hw.middleCols(ns + nl + nu, nz);
        const auto Hww = Hw.rightCols(nw);

        // Views to the sub-matrices in A = [As Al Au Az Aw Af]
        const auto Ax = A.leftCols(nx);
        const auto Al = Ax.middleCols(ns, nl);
        const auto Au = Ax.middleCols(ns + nl, nu);
        const auto Az = Ax.middleCols(ns + nl + nu, nz);
        const auto Aw = Ax.rightCols(nw);

        // Views to the sub-vectors in Z = [Zs Zl Zu Zz Zw Zf]
        const auto Zx = Z.head(nx);
        const auto Zs = Zx.head(ns);
        const auto Zl = Zx.segment(ns, nl);
        const auto Zu = Zx.segment(ns + nl, nu);
        const auto Zz = Zx.segment(ns + nl + nu, nz);
        const auto Zw = Zx.tail(nw);

        // Views to the sub-vectors in W = [Ws Wl Wu Wz Ww Wf]
        const auto Wx = W.head(nx);
        const auto Ws = Wx.head(ns);
        const auto Wl = Wx.segment(ns, nl);
        const auto Wu = Wx.segment(ns + nl, nu);
        const auto Wz = Wx.segment(ns + nl + nu, nz);
        const auto Ww = Wx.tail(nw);

        // Views to the sub-vectors in L = [Ls Ll Lu Lz Lw Lf]
        const auto Lx = L.head(nx);
        const auto Ls = Lx.head(ns);
        const auto Ll = Lx.segment(ns, nl);
        const auto Lu = Lx.segment(ns + nl, nu);
        const auto Lz = Lx.segment(ns + nl + nu, nz);
        const auto Lw = Lx.tail(nw);

        // Views to the sub-vectors in U = [Us Ul Uu Uz Uw Uf]
        const auto Ux = U.head(nx);
        const auto Us = Ux.head(ns);
        const auto Ul = Ux.segment(ns, nl);
        const auto Uu = Ux.segment(ns + nl, nu);
        const auto Uz = Ux.segment(ns + nl + nu, nz);
        const auto Uw = Ux.tail(nw);

        // The right-hand side vectors [a b c d]
        auto a = r.head(n);
        auto b = r.segment(n, m);
        auto c = r.segment(n + m, n);
        auto d = r.tail(n);

        // The solution vectors [x y z w]
        auto x = s.head(n);
        auto y = s.segment(n, m);
        auto z = s.segment(n + m, n);
        auto w = s.tail(n);

        // Views to the sub-vectors in a = [as al au az aw af]
        auto ax = a.head(nx);
        auto as = ax.head(ns);
        auto al = ax.segment(ns, nl);
        auto au = ax.segment(ns + nl, nu);
        auto az = ax.segment(ns + nl + nu, nz);
        auto aw = ax.tail(nw);
        auto af = a.tail(nf);

        // Views to the sub-vectors in c = [cs cl cu cz cw cf]
        auto cx = c.head(nx);
        auto cs = cx.head(ns);
        auto cl = cx.segment(ns, nl);
        auto cu = cx.segment(ns + nl, nu);
        auto cz = cx.segment(ns + nl + nu, nz);
        auto cw = cx.tail(nw);
        auto cf = c.tail(nf);

        // Views to the sub-vectors in d = [ds dl du dz dw df]
        auto dx = d.head(nx);
        auto ds = dx.head(ns);
        auto dl = dx.segment(ns, nl);
        auto du = dx.segment(ns + nl, nu);
        auto dz = dx.segment(ns + nl + nu, nz);
        auto dw = dx.tail(nw);
        auto df = d.tail(nf);

        // Views to the sub-vectors in x = [xs xl xu xz xw xf]
        auto xx = x.head(nx);
        auto xs = xx.head(ns);
        auto xl = xx.segment(ns, nl);
        auto xu = xx.segment(ns + nl, nu);
        auto xz = xx.segment(ns + nl + nu, nz);
        auto xw = xx.tail(nw);
        auto xf = x.tail(nf);

        // Views to the sub-vectors in z = [zs zl zu zz zw zf]
        auto zx = z.head(nx);
        auto zs = zx.head(ns);
        auto zl = zx.segment(ns, nl);
        auto zu = zx.segment(ns + nl, nu);
        auto zz = zx.segment(ns + nl + nu, nz);
        auto zw = zx.tail(nw);
        auto zf = z.tail(nf);

        // Views to the sub-vectors in w = [ws wl wu wz ww wf]
        auto wx = w.head(nx);
        auto ws = wx.head(ns);
        auto wl = wx.segment(ns, nl);
        auto wu = wx.segment(ns + nl, nu);
        auto wz = wx.segment(ns + nl + nu, nz);
        auto ww = wx.tail(nw);
        auto wf = w.tail(nf);

        // Initialize a, b, c, d in the ordering x = [xs, xl, xu, xf]
        a.noalias() = rhs.a(iordering);
        b.noalias() = rhs.b;
        c.noalias() = rhs.c(iordering);
        d.noalias() = rhs.d(iordering);

        // Calculate as', al', au', az', aw', b'
        as += cs/Ls + ds/Us - Hsl*(cl/Zl) - Hsu*(du/Wu) - Hsz*(cz/Zz) - Hsw*(dw/Ww);
        al += dl/Ul - Hll*(cl/Zl) - Hlu*(du/Wu) - Hlz*(cz/Zz) - Hlw*(dw/Ww) - (Wl/Ul) % (cl/Zl);
        au += cu/Lu - Hul*(cl/Zl) - Huu*(du/Wu) - Huz*(cz/Zz) - Huw*(dw/Ww) - (Zu/Lu) % (du/Wu);
        az += dz/Uz - Hzl*(cl/Zl) - Hzu*(du/Wu) - Hzz*(cz/Zz) - Hzw*(dw/Ww) - (Wz/Uz) % (cz/Zz);
        aw += cw/Lw - Hwl*(cl/Zl) - Hwu*(du/Wu) - Hwz*(cz/Zz) - Hww*(dw/Ww) - (Zw/Lw) % (dw/Ww);
         b -= Al*(cl/Zl) + Au*(du/Wu) + Az*(cz/Zz) + Aw*(dw/Ww);

        zz.noalias() = -az;
        ww.noalias() = -aw;

        az.fill(0.0);
        aw.fill(0.0);

        // Solve the saddle point problem
        res += spsolver.solve({a, b}, {x, y});

        // Calculate zz and ww
        zz.noalias() += Hzs*xs + Hzl*xl + Hzu*xu + tr(Az)*y;
        ww.noalias() += Hws*xs + Hwl*xl + Hwu*xu + tr(Aw)*y;

        // Calculate zl and wu
        zl.noalias() = -(Zl % xl)/Ll;
        wu.noalias() = -(Wu % xu)/Uu;

        // Calculate xl and xu
        xl.noalias() = (cl - Ll % zl)/Zl;
        xu.noalias() = (du - Uu % wu)/Wu;

        // Calculate xz and xw
        xz.noalias() = (cz - Lz % zz)/Zz;
        xw.noalias() = (dw - Uw % ww)/Ww;

        // Calculate zs and ws
        zs.noalias() = (cs - Zs % xs)/Ls;
        ws.noalias() = (ds - Ws % xs)/Us;

        // Calculate zu and zw
        zu.noalias() = (cu - Zu % xu)/Lu;
        zw.noalias() = (cw - Zw % xw)/Lw;

        // Calculate wl and wz
        wl.noalias() = (dl - Wl % xl)/Ul;
        wz.noalias() = (dz - Wz % xz)/Uz;

        // Calculate xf, zf, wf
        xf.noalias() = af;
        zf.noalias() = cf;
        wf.noalias() = df;

        // Permute the calculated (x z w) to their original order
        sol.y = y;
        sol.x(iordering) = x;
        sol.z(iordering) = z;
        sol.w(iordering) = w;

        return res.stop();
    }

    /// Solve the saddle point matrix equation with diagonal Hessian.
    auto solveDiagonalHessianMatrix(IpSaddlePointVector rhs, IpSaddlePointSolution sol) -> Result
    {
        // The result of this method call
        Result res;

        // Views to the blocks of the Hessian matrix Hxx
        const auto Hxx = H.diagonal.head(nx);
        const auto Hll = Hxx.segment(ns, nl);
        const auto Huu = Hxx.segment(ns + nl, nu);
        const auto Hzz = Hxx.segment(ns + nl + nu, nz);
        const auto Hww = Hxx.tail(nw);

        // Views to the sub-matrices in A = [As Al Au Az Aw Af]
        const auto Ax = A.leftCols(nx);
        const auto Al = Ax.middleCols(ns, nl);
        const auto Au = Ax.middleCols(ns + nl, nu);
        const auto Az = Ax.middleCols(ns + nl + nu, nz);
        const auto Aw = Ax.rightCols(nw);

        // Views to the sub-vectors in Z = [Zs Zl Zu Zz Zw Zf]
        const auto Zx = Z.head(nx);
        const auto Zs = Zx.head(ns);
        const auto Zl = Zx.segment(ns, nl);
        const auto Zu = Zx.segment(ns + nl, nu);
        const auto Zz = Zx.segment(ns + nl + nu, nz);
        const auto Zw = Zx.tail(nw);

        // Views to the sub-vectors in W = [Ws Wl Wu Wz Ww Wf]
        const auto Wx = W.head(nx);
        const auto Ws = Wx.head(ns);
        const auto Wl = Wx.segment(ns, nl);
        const auto Wu = Wx.segment(ns + nl, nu);
        const auto Wz = Wx.segment(ns + nl + nu, nz);
        const auto Ww = Wx.tail(nw);

        // Views to the sub-vectors in L = [Ls Ll Lu Lz Lw Lf]
        const auto Lx = L.head(nx);
        const auto Ls = Lx.head(ns);
        const auto Ll = Lx.segment(ns, nl);
        const auto Lu = Lx.segment(ns + nl, nu);
        const auto Lz = Lx.segment(ns + nl + nu, nz);
        const auto Lw = Lx.tail(nw);

        // Views to the sub-vectors in U = [Us Ul Uu Uz Uw Uf]
        const auto Ux = U.head(nx);
        const auto Us = Ux.head(ns);
        const auto Ul = Ux.segment(ns, nl);
        const auto Uu = Ux.segment(ns + nl, nu);
        const auto Uz = Ux.segment(ns + nl + nu, nz);
        const auto Uw = Ux.tail(nw);

        // The right-hand side vectors [a b c d]
        auto a = r.head(n);
        auto b = r.segment(n, m);
        auto c = r.segment(n + m, n);
        auto d = r.tail(n);

        // The solution vectors [x y z w]
        auto x = s.head(n);
        auto y = s.segment(n, m);
        auto z = s.segment(n + m, n);
        auto w = s.tail(n);

        // Views to the sub-vectors in a = [as al au az aw af]
        auto ax = a.head(nx);
        auto as = ax.head(ns);
        auto al = ax.segment(ns, nl);
        auto au = ax.segment(ns + nl, nu);
        auto az = ax.segment(ns + nl + nu, nz);
        auto aw = ax.tail(nw);
        auto af = a.tail(nf);

        // Views to the sub-vectors in c = [cs cl cu cz cw cf]
        auto cx = c.head(nx);
        auto cs = cx.head(ns);
        auto cl = cx.segment(ns, nl);
        auto cu = cx.segment(ns + nl, nu);
        auto cz = cx.segment(ns + nl + nu, nz);
        auto cw = cx.tail(nw);
        auto cf = c.tail(nf);

        // Views to the sub-vectors in d = [ds dl du dz dw df]
        auto dx = d.head(nx);
        auto ds = dx.head(ns);
        auto dl = dx.segment(ns, nl);
        auto du = dx.segment(ns + nl, nu);
        auto dz = dx.segment(ns + nl + nu, nz);
        auto dw = dx.tail(nw);
        auto df = d.tail(nf);

        // Views to the sub-vectors in x = [xs xl xu xz xw xf]
        auto xx = x.head(nx);
        auto xs = xx.head(ns);
        auto xl = xx.segment(ns, nl);
        auto xu = xx.segment(ns + nl, nu);
        auto xz = xx.segment(ns + nl + nu, nz);
        auto xw = xx.tail(nw);
        auto xf = x.tail(nf);

        // Views to the sub-vectors in z = [zs zl zu zz zw zf]
        auto zx = z.head(nx);
        auto zs = zx.head(ns);
        auto zl = zx.segment(ns, nl);
        auto zu = zx.segment(ns + nl, nu);
        auto zz = zx.segment(ns + nl + nu, nz);
        auto zw = zx.tail(nw);
        auto zf = z.tail(nf);

        // Views to the sub-vectors in w = [ws wl wu wz ww wf]
        auto wx = w.head(nx);
        auto ws = wx.head(ns);
        auto wl = wx.segment(ns, nl);
        auto wu = wx.segment(ns + nl, nu);
        auto wz = wx.segment(ns + nl + nu, nz);
        auto ww = wx.tail(nw);
        auto wf = w.tail(nf);

        // Initialize a, b, c, d in the ordering x = [xs, xl, xu, xf]
        a.noalias() = rhs.a(iordering);
        b.noalias() = rhs.b;
        c.noalias() = rhs.c(iordering);
        d.noalias() = rhs.d(iordering);

        // Calculate as', al', au', az', aw', b'
        as += cs/Ls + ds/Us;
        al += dl/Ul - Hll % (cl/Zl) - (Wl/Ul) % (cl/Zl);
        au += cu/Lu - Huu % (du/Wu) - (Zu/Lu) % (du/Wu);
        az += dz/Uz - Hzz % (cz/Zz) - (Wz/Uz) % (cz/Zz);
        aw += cw/Lw - Hww % (dw/Ww) - (Zw/Lw) % (dw/Ww);
         b -= Al*(cl/Zl) + Au*(du/Wu) + Az*(cz/Zz) + Aw*(dw/Ww);

        zz.noalias() = -az;
        ww.noalias() = -aw;

        az.fill(0.0);
        aw.fill(0.0);

        // Solve the saddle point problem
        res += spsolver.solve({a, b}, {x, y});

        // Calculate zz and ww
        zz.noalias() += tr(Az)*y;
        ww.noalias() += tr(Aw)*y;

        // Calculate zl and wu
        zl.noalias() = -(Zl % xl)/Ll;
        wu.noalias() = -(Wu % xu)/Uu;

        // Calculate xl and xu
        xl.noalias() = (cl - Ll % zl)/Zl;
        xu.noalias() = (du - Uu % wu)/Wu;

        // Calculate xz and xw
        xz.noalias() = (cz - Lz % zz)/Zz;
        xw.noalias() = (dw - Uw % ww)/Ww;

        // Calculate zs and ws
        zs.noalias() = (cs - Zs % xs)/Ls;
        ws.noalias() = (ds - Ws % xs)/Us;

        // Calculate zu and zw
        zu.noalias() = (cu - Zu % xu)/Lu;
        zw.noalias() = (cw - Zw % xw)/Lw;

        // Calculate wl and wz
        wl.noalias() = (dl - Wl % xl)/Ul;
        wz.noalias() = (dz - Wz % xz)/Uz;

        // Calculate xf, zf, wf
        xf.noalias() = af;
        zf.noalias() = cf;
        wf.noalias() = df;

        // Permute the calculated (x z w) to their original order
        sol.y = y;
        sol.x(iordering) = x;
        sol.z(iordering) = z;
        sol.w(iordering) = w;

        return res.stop();
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
    pimpl->spsolver.setOptions(options);
}

auto IpSaddlePointSolver::options() const -> const SaddlePointOptions&
{
    return pimpl->spsolver.options();
}

auto IpSaddlePointSolver::initialize(MatrixConstRef A) -> Result
{
    return pimpl->initialize(A);
}

auto IpSaddlePointSolver::decompose(IpSaddlePointMatrix lhs) -> Result
{
    return pimpl->decompose(lhs);
}

auto IpSaddlePointSolver::solve(IpSaddlePointVector rhs, IpSaddlePointSolution sol) -> Result
{
    return pimpl->solve(rhs, sol);
}

} // namespace Optima
