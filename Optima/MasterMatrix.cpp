// Optima is a C++ library for solving linear and non-linear constrained optimization problems
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
// GNU General Public License for more Mbar.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "MasterMatrix.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/Utils.hpp>
#include <Optima/MasterMatrixH.hpp>
#include <Optima/MasterMatrixV.hpp>
#include <Optima/MasterMatrixW.hpp>

namespace Optima {

struct MasterMatrix::Impl
{
    const Index nx = 0;   ///< The number of variables x.
    const Index np = 0;   ///< The number of variables p.
    const Index ny = 0;   ///< The number of variables y.
    const Index nz = 0;   ///< The number of variables z.
    const Index nw = 0;   ///< The number of variables w = (y, z).

    Index ns = 0;   ///< The number of stable variables.
    Index nu = 0;   ///< The number of unstable variables.

    Index nb = 0;   ///< The number of basic variables.
    Index nn = 0;   ///< The number of non-basic variables.
    Index nl = 0;   ///< The number of linearly dependent rows in Wx = [Ax; Jx].

    Matrix R;        ///< The matrix R in R*Wx*Q = [Ibb Sbn].
    Matrix S;        ///< The matrix S' = [Sbsns Sbsnu Sbsp; 0 Sbunu Sbup].
    Indices jbn;     ///< The order of x variables as x = (xb, xn) = (xbs, xbu, xns, xnu) = (xbe, xbi, xbu, xne, xni, xnu).

    Index nbs = 0;  ///< The number of basic stable variables.
    Index nbu = 0;  ///< The number of basic unstable variables.
    Index nns = 0;  ///< The number of non-basic stable variables.
    Index nnu = 0;  ///< The number of non-basic unstable variables.

    Index nbe = 0;  ///< The number of explicit basic stable variables.
    Index nbi = 0;  ///< The number of implicit basic stable variables.
    Index nne = 0;  ///< The number of explicit non-basic stable variables.
    Index nni = 0;  ///< The number of implicit non-basic stable variables.

    Indices bs;      ///< The boolean flags that indicate which variables in x are stable.
    Indices Kb;      ///< The permutation matrix used to order the basic variables as xb = (xbe, xbi, xbu) with `e` and `i` denoting pivot and non-pivot.
    Indices Kn;      ///< The permutation matrix used to order the non-basic variables as xn = (xne, xni, xnu) with `e` and `i` denoting pivot and non-pivot.

    Indices jsu;     ///< The order of x variables as x = (xs, xu) = (xbs, xns, xbu, xnu) = (xbe, xbi, xne, xni, xbu, xnu).

    Matrix Hprime;   ///< The matrix H' = [Hss Hsp].
    Matrix Vprime;   ///< The matrix V' = [Vps Vpu Vpp].
    Matrix Wprime;   ///< The matrix W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].

    Impl(Index nx, Index np, Index ny, Index nz)
    : nx(nx), np(np), ny(ny), nz(nz), nw(ny + nz)
    {
        S.resize(nw, nx + np);
        Hprime.resize(nx, nx + np);
        Vprime.resize(np, nx + np);
        Wprime.resize(nw, nx + np);
        jbn.resize(nx);
        jsu.resize(nx);
    }

    auto update(const MasterMatrixH& H, const MasterMatrixV& V, const MasterMatrixW& W, IndicesConstRef ju0) -> void
    {
        const auto Wbar = W.echelonForm();

        //======================================================================
        // Initialize number of basic/non-basic stable/unstable variables
        //======================================================================

        ns = nx - ju0.size();
        nu = ju0.size();

        nb = Wbar.jb.size();
        nn = Wbar.jn.size();
        nl = nw - nb;

        //======================================================================
        // Initialize matrices R, Sbn, Sbp and indices of variables jbn
        //======================================================================

        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        R = Wbar.R;

        Sbn = Wbar.Sbn;
        Sbp = Wbar.Sbp;

        jbn << Wbar.jb, Wbar.jn; // the indices of the variables ordered as x = (xb, xn), but xb and xn not yet properly sorted

        auto jb = jbn.head(nb); // the indices of the basic variables
        auto jn = jbn.tail(nn); // the indices of the non-basic variables

        //======================================================================
        // Initialize permutation matrices Kb and Kn so that
        // basic and non-basic variables can later be ordered as:
        //     xb = (xbs, xbu) = (xbe, xbi, xbu)
        //     xn = (xns, xnu) = (xne, xni, xnu)
        //======================================================================

        Kb = indices(nb);
        Kn = indices(nn);

        bs.setOnes(nx); // 1 for stable, 0 for unstable
        bs(ju0).fill(0);

        const auto Hd = H.Hxx.diagonal(); // the diagonal entries in Hxx used to sort the variables

        using std::abs;
        using std::sort;

        auto jb_kth_is_stable = [&](auto i) { return bs[jb[i]]; }; // returts true if k-th basic variable is stable
        auto jn_kth_is_stable = [&](auto i) { return bs[jn[i]]; }; // returts true if k-th non-basic variable is stable

        // Partition Kb = (Kbs, Kbu) and Kn = (Kns, Knu)
        nbs = moveLeftIf(Kb, jb_kth_is_stable); // as a result, update the number of basic stable variables
        nns = moveLeftIf(Kn, jn_kth_is_stable); // as a result, update the number of non-basic stable variables

        nbu = nb - nbs; // update the number of basic unstable variables
        nnu = nn - nns; // update the number of non-basic unstable variables

        // Ensure no basic variable has been marked as unstable.
        error(nbu > 0, "MasterMatrix::update failed with given indices of "
            "unstable variables, which contain indices of basic variables.");

        // Partition Kbs = (Kbe, Kbi) and Kns = (Kne, Kni)
        auto Kbs = Kb.head(nbs);
        auto Kns = Kn.head(nns);

        // Return true if the k-th stable basic variable is a pivot/explicit variable
        auto jbs_kth_is_explicit = [&](auto k)
        {
            const auto idx = jb[k];                     // the global index of the k-th basic variable
            const auto Hkk = Hd[idx];                   // the corresponding diagonal entry in the H matrix
            const auto a1 = 1.0;                        // the max value along the corresponding column of the identity matrix
            const auto a2 = norminf(V.Vpx.col(idx));    // the max value along the corresponding column of the Vpx matrix
            return abs(Hkk) >= std::max(a1, a2);        // return true if diagonal entry is dominant with respect to Vpx and Ibb only (not Hxx!)
        };

        // Return true if the k-th stable non-basic variable is a pivot/explicit variable
        auto jns_kth_is_explicit = [&](auto k)
        {
            const auto idx = jn[k];                     // the global index of the k-th non-basic variable
            const auto Hkk = Hd[idx];                   // the corresponding diagonal entry in the H matrix
            const auto a1 = norminf(Sbn.col(k));        // the max value along the corresponding column of the Sbn matrix
            const auto a2 = norminf(V.Vpx.col(idx));    // the max value along the corresponding column of the Vpx matrix
            return abs(Hkk) >= std::max(a1, a2);        // return true if diagonal entry is dominant with respect to Vpx and Sbn only (not Hxx!)
        };

        nbe = moveLeftIf(Kbs, jbs_kth_is_explicit); // as a result, update the number of basic stable explicit variables
        nne = moveLeftIf(Kns, jns_kth_is_explicit); // as a result, update the number of non-basic stable explicit variables

        // Update the number of non-pivot/implicit stable basic and non-basic variables.
        nbi = nbs - nbe;
        nni = nns - nne;

        //======================================================================
        // Apply permutation to Rb, Sbn, Sbp and indices of variables jbn
        //======================================================================
        auto Rb = R.topRows(nb);

        Kb.asPermutation().transpose().applyThisOnTheLeft(Sbn);
        Kb.asPermutation().transpose().applyThisOnTheLeft(Sbp);

        Kn.asPermutation().applyThisOnTheRight(Sbn);

        Kb.asPermutation().transpose().applyThisOnTheLeft(Rb);

        Kb.asPermutation().transpose().applyThisOnTheLeft(jb); // jb is now ordered as (jbs, jbu) = (jbe, jbi, jbu)
        Kn.asPermutation().transpose().applyThisOnTheLeft(jn); // jn is now ordered as (jns, jnu) = (jne, jni, jnu)

        // ---------------------------------------------------------------------
        // ***NOTE***
        // ---------------------------------------------------------------------
        // At this point, Sbn and Sbp have now the following structure:
        //     Sbn = [Sbsns Sbsnu; 0 Sbunu; 0bl]
        //     Sbp = [Sbsp; Sbup; 0bl]
        // ---------------------------------------------------------------------

        //=========================================================================================
        // Update the order of x variables as x = (xs, xu) = (xbs, xns, xbu, xnu), where:
        // -- xbs are basic xs variables;
        // -- xns are non-basic xs variables;
        // -- xbu are basic xu variables;
        // -- xnu are non-basic xu variables.
        //-----------------------------------------------------------------------------------------
        // Note: By moving unstable-variables away, we now have:
        // -- xbs = (xbe, xbi);
        // -- xns = (xne, xni).
        //=========================================================================================

        // The indices of the basic variables in xs and xu (jbs and jbu respectively)
        const auto jbs = jb.head(nbs);
        const auto jbu = jb.tail(nbu);

        // The indices of the non-basic variables in xs and xu (jns and jnu respectively)
        const auto jns = jn.head(nns);
        const auto jnu = jn.tail(nnu);

        // Initialize jsu = (js, ju) = (jbs, jns, jbu, jnu)
        jsu << jbs, jns, jbu, jnu;

        // The indices of the stable variables js = (jbs, jns) = (jbe, jbi, jne, jni)
        const auto js = jsu.head(ns);
        const auto ju = jsu.tail(nu);

        //=========================================================================================
        // Initialize matrices Hss, Hsp
        //=========================================================================================
        auto Hss = Hprime.topLeftCorner(ns, ns);
        auto Hsp = Hprime.topRightCorner(ns, np);

        Hss = H.Hxx(js, js);
        Hsp = H.Hxp(js, Eigen::all);

        //=========================================================================================
        // Initialize matrices Vps, Vpp
        //=========================================================================================
        auto Vps = Vprime.topLeftCorner(np, ns);
        auto Vpp = Vprime.topRightCorner(np, np);

        Vps = V.Vpx(Eigen::all, js);
        Vpp = V.Vpp;

        //=========================================================================================
        // Initialize matrices Ws, Wu and Wp in W = [Ws Wu Wp] = [As Au Ap; Js Ju Jp]
        //=========================================================================================

        auto Ws = Wprime.leftCols(ns);
        auto Wu = Wprime.middleCols(ns, nu);
        auto Wp = Wprime.rightCols(np);

        Ws = W.Wx(Eigen::all, js);
        Wu = W.Wx(Eigen::all, ju);
        Wp = W.Wp;
    }

    auto canonicalMatrix() const -> CanonicalMatrix
    {
        const auto dims = CanonicalDims{nx, np, ny, nz, nw, ns, nu, nb, nn, nl, nbs, nbu, nns, nnu, nbe, nbi, nne, nni};
        const auto Hss = Hprime.topLeftCorner(ns, ns);
        const auto Hsp = Hprime.topRightCorner(ns, np);
        const auto Vps = Vprime.topLeftCorner(np, ns);
        const auto Vpp = Vprime.topRightCorner(np, np);
        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp = S.topRightCorner(nbs, np);
        const auto Ws = Wprime.leftCols(ns);
        const auto Wu = Wprime.middleCols(ns, nu);
        const auto Wp = Wprime.rightCols(np);
        const auto As = Ws.topRows(ny);
        const auto Au = Wu.topRows(ny);
        const auto Ap = Wp.topRows(ny);
        const auto Js = Ws.bottomRows(nz);
        const auto Ju = Wu.bottomRows(nz);
        const auto Jp = Wp.bottomRows(nz);
        const auto jb = jbn.head(nb);
        const auto jn = jbn.tail(nn);
        const auto js = jsu.head(ns);
        const auto ju = jsu.tail(nu);

        return {dims, Hss, Hsp, Vps, Vpp, Sbsns, Sbsp};
    }

    auto canonicalForm() const -> CanonicalDetails
    {
        const auto dims = CanonicalDims{nx, np, ny, nz, nw, ns, nu, nb, nn, nl, nbs, nbu, nns, nnu, nbe, nbi, nne, nni};
        const auto Hss = Hprime.topLeftCorner(ns, ns);
        const auto Hsp = Hprime.topRightCorner(ns, np);
        const auto Vps = Vprime.topLeftCorner(np, ns);
        const auto Vpp = Vprime.topRightCorner(np, np);
        const auto Sbn = S.topLeftCorner(nb, nn);
        const auto Sbp = S.topRightCorner(nb, np);
        const auto Ws = Wprime.leftCols(ns);
        const auto Wu = Wprime.middleCols(ns, nu);
        const auto Wp = Wprime.rightCols(np);
        const auto As = Ws.topRows(ny);
        const auto Au = Wu.topRows(ny);
        const auto Ap = Wp.topRows(ny);
        const auto Js = Ws.bottomRows(nz);
        const auto Ju = Wu.bottomRows(nz);
        const auto Jp = Wp.bottomRows(nz);
        const auto jb = jbn.head(nb);
        const auto jn = jbn.tail(nn);
        const auto js = jsu.head(ns);
        const auto ju = jsu.tail(nu);

        return {dims, Hss, Hsp, Vps, Vpp, Sbn, Sbp, R, Ws, Wu, Wp, As, Au, Ap, Js, Ju, Jp, jb, jn, js, ju};
    }
};

MasterMatrix::MasterMatrix(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz))
{}

MasterMatrix::MasterMatrix(const MasterMatrix& other)
: pimpl(new Impl(*other.pimpl))
{}

MasterMatrix::~MasterMatrix()
{}

auto MasterMatrix::operator=(MasterMatrix other) -> MasterMatrix&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto MasterMatrix::update(const MasterMatrixH& H, const MasterMatrixV& V, const MasterMatrixW& W, IndicesConstRef ju) -> void
{
    pimpl->update(H, V, W, ju);
}

auto MasterMatrix::canonicalMatrix() const -> CanonicalMatrix
{
    return pimpl->canonicalMatrix();
}

auto MasterMatrix::canonicalForm() const -> CanonicalDetails
{
    return pimpl->canonicalForm();
}

auto MasterMatrix::matrix() const -> Matrix
{
    const auto Mbar = canonicalForm();
    const auto dims = Mbar.dims;

    const auto nx = dims.nx;
    const auto np = dims.np;
    const auto ny = dims.ny;
    const auto nz = dims.nz;
    const auto nw = dims.nw;
    const auto ns = dims.ns;
    const auto nu = dims.nu;

    const auto t = nx + np + ny + nz;

    const auto Hss = Mbar.Hss;
    const auto Hsp = Mbar.Hsp;
    const auto Vps = Mbar.Vps;
    const auto Vpp = Mbar.Vpp;
    const auto Ws = Mbar.Ws;
    const auto Wp = Mbar.Wp;
    const auto js = Mbar.js;
    const auto ju = Mbar.ju;

    Matrix mat = zeros(t, t);

    auto matHxx = mat.topRows(nx).leftCols(nx);
    auto matHxp = mat.topRows(nx).middleCols(nx, np);
    auto matWxT = mat.topRows(nx).rightCols(nw);
    auto matVpx = mat.middleRows(nx, np).leftCols(nx);
    auto matVpp = mat.middleRows(nx, np).middleCols(nx, np);
    auto matWx  = mat.bottomRows(nw).leftCols(nx);
    auto matWp  = mat.bottomRows(nw).middleCols(nx, np);

    using Eigen::all;

    const auto Iuu = identity(nu, nu);

    matHxx(js, js) = Hss;
    matHxx(ju, ju) = Iuu;
    matHxp(js, all) = Hsp;
    matWxT(js, all) = tr(Ws);
    matVpx(all, js) = Vps;
    matVpp = Vpp;
    matWx(all, js) = Ws;
    matWp = Wp;

    return mat;
}

} // namespace Optima
