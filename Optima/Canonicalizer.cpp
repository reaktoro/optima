// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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

#include "Canonicalizer.hpp"

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>

namespace Optima {

struct Canonicalizer::Impl
{
    MasterDims dims;    ///< The dimensions of the variables x, p, y, z, w.

    Index ns = 0;       ///< The number of stable variables.
    Index nu = 0;       ///< The number of unstable variables.

    Index nb = 0;       ///< The number of basic variables.
    Index nn = 0;       ///< The number of non-basic variables.
    Index nl = 0;       ///< The number of linearly dependent rows in Wx = [Ax; Jx].

    Matrix R;           ///< The matrix R in RWQ = [Ibb Sbn Sbp].
    Matrix S;           ///< The matrix S' = [Sbsns Sbsnu Sbsp].
    Indices jbn;        ///< The order of x variables as x = (xb, xn) = (xbs, xns, xnu) = (xbe, xbi, xne, xni, xnu).

    Index nbs = 0;      ///< The number of basic stable variables.
    Index nbu = 0;      ///< The number of basic unstable variables.
    Index nns = 0;      ///< The number of non-basic stable variables.
    Index nnu = 0;      ///< The number of non-basic unstable variables.

    Index nbe = 0;      ///< The number of explicit basic stable variables.
    Index nbi = 0;      ///< The number of implicit basic stable variables.
    Index nne = 0;      ///< The number of explicit non-basic stable variables.
    Index nni = 0;      ///< The number of implicit non-basic stable variables.

    Indices bs;         ///< The boolean flags that indicate which variables in x are stable.
    Indices Kb;         ///< The permutation matrix used to order the basic variables as xb = (xbe, xbi) with `e` and `i` denoting pivot and non-pivot.
    Indices Kn;         ///< The permutation matrix used to order the non-basic variables as xn = (xne, xni, xnu) with `e` and `i` denoting pivot and non-pivot.

    Indices jsu;        ///< The order of x variables as x = (xbs, xns, xnu) = (xbe, xbi, xne, xni, xnu).

    Matrix Hprime;      ///< The matrix H' = [Hss Hsp].
    Matrix Vprime;      ///< The matrix V' = [Vps Vpu Vpp].
    Matrix Wprime;      ///< The matrix W' = [Ws Wu Wp] = [As Au Ap; Js Ju Jp].

    bool diagHxx = false; ///< The flag indicating whether Hxx is diagonal.

    Impl()
    {}

    Impl(const MasterMatrix& M)
    {
        update(M);
    }

    auto update(const MasterMatrix& M) -> void
    {
        dims = M.dims;

        const auto [nx, np, ny, nz, nw, nt] = dims;

        const auto H   = M.H;
        const auto V   = M.V;
        const auto W   = M.W;
        const auto RWQ = M.RWQ;
        const auto ju0 = M.ju;

        //======================================================================
        // Initialize number of basic/non-basic stable/unstable variables
        //======================================================================

        ns = nx - ju0.size();
        nu = ju0.size();

        nb = RWQ.jb.size();
        nn = RWQ.jn.size();
        nl = nw - nb;

        //======================================================================
        // Initialize matrices R, Sbn, Sbp and indices of variables jbn
        //======================================================================

        S.resize(nw, nx + np);
        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        R = RWQ.R;

        Sbn = RWQ.Sbn;
        Sbp = RWQ.Sbp;

        jbn.resize(nx);
        jbn << RWQ.jb, RWQ.jn; // the indices of the variables ordered as x = (xb, xn), but xb and xn not yet properly sorted

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
        error(nbu > 0, "Canonicalizer::update failed with given indices of "
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
        // Apply permutation to R, Sbn, Sbp and indices of variables jbn
        //======================================================================
        Kb.asPermutation().transpose().applyThisOnTheLeft(Sbn);
        Kb.asPermutation().transpose().applyThisOnTheLeft(Sbp);

        Kn.asPermutation().applyThisOnTheRight(Sbn);

        Kb.asPermutation().transpose().applyThisOnTheLeft(R);

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
        const auto jbs = jb.head(nbs); // nbs === nb
        const auto jbu = jb.tail(nbu); // nbu === 0

        // The indices of the non-basic variables in xs and xu (jns and jnu respectively)
        const auto jns = jn.head(nns);
        const auto jnu = jn.tail(nnu);

        // Initialize jsu = (jbs, jns, jnu)
        jsu.resize(nx);
        jsu << jbs, jns, jnu;

        // The indices of the stable variables js = (jbs, jns) = (jbe, jbi, jne, jni)
        const auto js = jsu.head(ns);
        const auto ju = jsu.tail(nu);

        //=========================================================================================
        // Initialize matrices Hss, Hsp
        //=========================================================================================
        using Eigen::all;

        Hprime.resize(nx, nx + np);
        auto Hss = Hprime.topLeftCorner(ns, ns);
        auto Hsp = Hprime.topRightCorner(ns, np);

        Hss = H.Hxx(js, js);
        Hsp = H.Hxp(js, all);

        diagHxx = H.isHxxDiag;

        //=========================================================================================
        // Initialize matrices Vps, Vpp
        //=========================================================================================
        Vprime.resize(np, nx + np);
        auto Vps = Vprime.topLeftCorner(np, ns);
        auto Vpp = Vprime.topRightCorner(np, np);

        Vps = V.Vpx(all, js);
        Vpp = V.Vpp;

        //=========================================================================================
        // Initialize matrices Ws, Wu and Wp in W = [Ws Wu Wp] = [As Au Ap; Js Ju Jp]
        //=========================================================================================

        Wprime.resize(nw, nx + np);
        auto Ws = Wprime.leftCols(ns);
        auto Wu = Wprime.middleCols(ns, nu);
        auto Wp = Wprime.rightCols(np);

        Ws = W.Wx(all, js);
        Wu = W.Wx(all, ju);
        Wp = W.Wp;
    }

    auto canonicalMatrix() const -> CanonicalMatrix
    {
        const auto [nx, np, ny, nz, nw, nt] = dims;

        const auto dims = CanonicalDims{nx, np, ny, nz, nw, nt, ns, nu, nb, nn, nl, nbs, nbu, nns, nnu, nbe, nbi, nne, nni};
        const auto Hss = Hprime.topLeftCorner(ns, ns);
        const auto Hsp = Hprime.topRightCorner(ns, np);
        const auto Vps = Vprime.topLeftCorner(np, ns);
        const auto Vpp = Vprime.topRightCorner(np, np);
        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp = S.topRightCorner(nbs, np);
        const auto Rbs = R.topRows(nbs);
        const auto jb = jbn.head(nb);
        const auto jn = jbn.tail(nn);
        const auto js = jsu.head(ns);
        const auto ju = jsu.tail(nu);

        return {dims, Hss, Hsp, Vps, Vpp, Sbsns, Sbsp, Rbs, jb, jn, js, ju};
    }
};

Canonicalizer::Canonicalizer()
: pimpl(new Impl())
{}

Canonicalizer::Canonicalizer(const MasterMatrix& M)
: pimpl(new Impl(M))
{}

Canonicalizer::Canonicalizer(const Canonicalizer& other)
: pimpl(new Impl(*other.pimpl))
{}

Canonicalizer::~Canonicalizer()
{}

auto Canonicalizer::operator=(Canonicalizer other) -> Canonicalizer&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Canonicalizer::update(const MasterMatrix& M) -> void
{
    pimpl->update(M);
}

auto Canonicalizer::canonicalMatrix() const -> CanonicalMatrix
{
    return pimpl->canonicalMatrix();
}

} // namespace Optima
