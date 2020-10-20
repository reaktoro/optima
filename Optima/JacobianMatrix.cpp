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
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "JacobianMatrix.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/SaddlePointTypes.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct JacobianMatrix::Impl
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
    }

    auto update(const JacobianBlockH& H, const JacobianBlockV& V, const JacobianBlockW& W, IndicesConstRef ju) -> void
    {
        const auto Wbar = W.canonicalForm();

        //======================================================================
        // Initialize number of basic/non-basic stable/unstable variables
        //======================================================================

        ns = nx - ju.size();
        nu = ju.size();

        nb = Wbar.jb.size();
        nn = Wbar.jn.size();
        nl = nw - nb;

        //======================================================================
        // Initialize matrices R, Sbn, Sbp and indices of variables jbn
        //======================================================================

        auto Rb = R.topRows(nb);

        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        auto jb = jbn.head(nb); // the indices of the basic variables
        auto jn = jbn.tail(nn); // the indices of the non-basic variables

        R = Wbar.R;

        Sbn = Wbar.Sbn;
        Sbp = Wbar.Sbp;

        jbn << Wbar.jb, Wbar.jn; // the indices of the variables ordered as x = (xb, xn), but xb and xn not yet properly sorted

        //======================================================================
        // Initialize permutation matrices Kb and Kn so that
        // basic and non-basic variables can later be ordered as:
        //     xb = (xbs, xbu) = (xbe, xbi, xbu)
        //     xn = (xns, xnu) = (xne, xni, xnu)
        //======================================================================

        Kb = indices(nb);
        Kn = indices(nn);

        bs.setOnes(nx); // 1 for stable, 0 for unstable
        bs(ju).fill(0);

        auto is_stable_basic = [&](auto i) { return bs[jb[i]]; }; // returts true if ith basic variable is stable
        auto is_stable_nonbasic = [&](auto i) { return bs[jn[i]]; }; // returts true if ith non-basic variable is stable

        // Partition Kb = (Kbs, Kbu) and Kn = (Kns, Knu)
        nbs = moveLeftIf(Kb, is_stable_basic);    // as a result, update the number of basic stable variables
        nns = moveLeftIf(Kn, is_stable_nonbasic); // as a result, update the number of non-basic stable variables

        nbu = nb - nbs; // update the number of basic unstable variables
        nnu = nn - nns; // update the number of non-basic unstable variables

        // Sort Kbs and Kns in descending order of diagonal values in Hxx
        auto Kbs = Kb.head(nbs);
        auto Kns = Kn.head(nns);

        const auto Hd = H.Hxx.diagonal(); // the diagonal entries in Hxx used to sort the variables

        using std::abs;
        using std::sort;

        sort(Kbs.begin(), Kbs.end(), [&](auto l, auto r) { return abs(Hd[jb[l]]) > abs(Hd[jb[r]]); });
        sort(Kns.begin(), Kns.end(), [&](auto l, auto r) { return abs(Hd[jn[l]]) > abs(Hd[jn[r]]); });

        //======================================================================
        // Apply permutation to R, Sbn, Sbp and indices of variables jbn
        //======================================================================

        Kb.asPermutation().transpose().applyThisOnTheLeft(Sbn);
        Kb.asPermutation().transpose().applyThisOnTheLeft(Sbp);

        Kn.asPermutation().applyThisOnTheRight(Sbn);

        Kb.asPermutation().transpose().applyThisOnTheLeft(Rb);

        Kb.asPermutation().transpose().applyThisOnTheLeft(jb); // jb is now ordered as (jbs, jbu) = (jbe, jbi, jbu)
        Kn.asPermutation().transpose().applyThisOnTheLeft(jn); // jn is now ordered as (jns, jnu) = (jne, jni, jnu)

        //---------------------------------------------------------------------
        // ***NOTE***
        //---------------------------------------------------------------------
        // At this point, Sbn and Sbp have now the following structure:
        //     Sbn = [Sbsns Sbsnu; 0 Sbunu; 0bl]
        //     Sbp = [Sbsp; Sbup; 0bl]
        //---------------------------------------------------------------------

        //=========================================================================================
        // Identify the explicit/implicit basic/non-basic variables
        //=========================================================================================

        // Return true if the i-th basic variable is a pivot/explicit variable
        const auto is_basic_explicit = [&](auto i)
        {
            const auto idx = jb[i];                     // the global index of the basic variable
            const auto Hii = Hd[idx];                   // the corresponding diagonal entry in the H matrix
            const auto a1 = 1.0;                        // the max value along the corresponding column of the identity matrix
            const auto a2 = norminf(V.Vpx.col(idx));    // the max value along the corresponding column of the Vpx matrix
            return abs(Hii) >= std::max(a1, a2);        // return true if diagonal entry is dominant with respect to Vpx and Ibb only (not Hxx!)
        };

        // Return true if the i-th non-basic variable is a pivot/explicit variable
        const auto is_nonbasic_explicit = [&](auto i)
        {
            const auto idx = jn[i];                     // the global index of the non-basic variable
            const auto Hii = Hd[idx];                   // the corresponding diagonal entry in the H matrix
            const auto a1 = norminf(Sbn.col(i));        // the max value along the corresponding column of the Sbn matrix
            const auto a2 = norminf(V.Vpx.col(idx));    // the max value along the corresponding column of the Vpx matrix
            return abs(Hii) >= std::max(a1, a2);        // return true if diagonal entry is dominant with respect to Vpx and Sbn only (not Hxx!)
        };

        // Find the number of pivot/explicit basic variables (those with |Hbebe| >= I and |Hbebe| >= |cols(H, jbe)|)
        // Walk from first to last stable basic variable, since they are ordered in decresiang order of |Hii| values
        nbe = 0; while(nbe < nbs && is_basic_explicit(nbe)) ++nbe;

        // Find the number of pivot/explicit non-basic variables (those with |Hnene| >= |Sbsne| and |Hnene| > |cols(H, jne)|)
        // Walk from first to last stable non-basic variable, since they are ordered in decresiang order of |Hii| values
        nne = 0; while(nne < nns && is_nonbasic_explicit(nne)) ++nne;

        // Update the number of non-pivot/implicit stable basic and non-basic variables.
        nbi = nbs - nbe;
        nni = nns - nne;

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

    auto dims() const -> Dims
    {
        return {nx, np, ny, nz, nw, ns, nu, nb, nn, nl, nbs, nbu, nns, nnu, nbe, nbi, nne, nni};
    }

    auto canonicalForm() const -> CanonicalForm
    {
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

        return {Hss, Hsp, Vps, Vpp, Sbsns, Sbsp, R, Ws, Wu, Wp, As, Au, Ap, Js, Ju, Jp, jb, jn, js, ju};
    }
};

JacobianMatrix::JacobianMatrix(Index nx, Index np, Index ny, Index nz)
: pimpl(new Impl(nx, np, ny, nz))
{}

JacobianMatrix::JacobianMatrix(const JacobianMatrix& other)
: pimpl(new Impl(*other.pimpl))
{}

JacobianMatrix::~JacobianMatrix()
{}

auto JacobianMatrix::operator=(JacobianMatrix other) -> JacobianMatrix&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto JacobianMatrix::update(const JacobianBlockH& H, const JacobianBlockV& V, const JacobianBlockW& W, IndicesConstRef ju) -> void
{
    pimpl->update(H, V, W, ju);
}

auto JacobianMatrix::dims() const -> Dims
{
    return pimpl->dims();
}

auto JacobianMatrix::canonicalForm() const -> CanonicalForm
{
    return pimpl->canonicalForm();
}

} // namespace Optima
