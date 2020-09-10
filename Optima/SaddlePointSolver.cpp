// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
//
// This program is stable software: you can redistribute it and/or modify
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

#include "SaddlePointSolver.hpp"

// C++ includes
#include <cassert>
#include <cmath>

// Optima includes
#include <Optima/Canonicalizer.hpp>
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/Macros.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/SaddlePointSolverFullspace.hpp>
#include <Optima/SaddlePointSolverNullspace.hpp>
#include <Optima/SaddlePointSolverRangespace.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct SaddlePointSolver::Impl
{
    Matrix Ax;                              ///< The matrix block Ax of the saddle point problem.
    Matrix Ap;                              ///< The matrix block Ap of the saddle point problem.
    Canonicalizer canonicalizer;            ///< The canonicalizer of matrix *Ax*.
    SaddlePointSolverRangespace rangespace; ///< The canonical saddle point solver based on a rangespace algorithm.
    SaddlePointSolverNullspace nullspace;   ///< The canonical saddle point solver based on a nullspace algorithm.
    SaddlePointSolverFullspace fullspace;   ///< The canonical saddle point solver based on a fullspace algorithm.
    SaddlePointOptions options;             ///< The options used to solve the saddle point problems.
    SaddlePointDims dims;                   ///< The dimensions of the saddle point problem.
    Matrix S;                               ///< The S = [Sbsns Sbsnu Sbsp; 0 Sbunu Sbup] matrix that stores the canonical form of A = [Ax Ap].
    Matrix Hw;                              ///< The workspace for H matrix.
    Matrix Vw;                              ///< The workspace for V matrix.
    Matrix Jw;                              ///< The workspace for J matrix.
    Vector gw;                              ///< The workspace for right-hand side vector g
    Vector axw;                             ///< The workspace for right-hand side vector ax
    Vector apw;                             ///< The workspace for right-hand side vector ap
    Vector ayw;                             ///< The workspace for right-hand side vector ay
    Vector azw;                             ///< The workspace for right-hand side vector az
    Vector bw;                              ///< The workspace for right-hand side vector b
    Vector xw;                              ///< The workspace for solution vector x
    Vector pw;                              ///< The workspace for solution vector p
    Vector yw;                              ///< The workspace for solution vector y
    Vector zw;                              ///< The workspace for solution vector z
    Vector weights;                         ///< The priority weights for the selection of basic variables.
    Indices jx;                             ///< The order of x variables as x = (xbs, xns, xbu, xnu).
    Indices Kb;                             ///< The permutation matrix used to order the basic variables as xb = (xbe, xbi, xbu) with `e` and `i` denoting pivot and non-pivot
    Indices Kn;                             ///< The permutation matrix used to order the non-basic variables as xn = (xne, xni, xnu) with `e` and `i` denoting pivot and non-pivot
    bool degenerate = false;                ///< The boolean flag that indicates that the decomposed saddle point matrix was degenerate with no stable variables.

    /// Construct a SaddlePointSolver::Impl instance with given data.
    Impl(SaddlePointSolverInitArgs args)
    : Ax(args.Ax), Ap(args.Ap), canonicalizer(args.Ax),
      rangespace(args.nx, args.np, args.ny, args.nz),
      nullspace(args.nx, args.np, args.ny, args.nz),
      fullspace(args.nx, args.np, args.ny, args.nz)
    {
        // Ensure consistent and proper dimensions
        assert(args.nx > 0);
        assert(args.Ax.rows() == 0 || args.Ax.rows() <= args.ny);
        assert(args.Ap.rows() == 0 || args.Ap.rows() <= args.ny);
        assert(args.Ax.rows() == 0 || args.Ax.cols() == args.nx);
        assert(args.Ap.rows() == 0 || args.Ap.cols() == args.np);

        // Initialize the number of variables x, xs, xu, p, y, z
        const auto nx = dims.nx = args.nx;
        const auto ns = dims.ns = nx;
        const auto nu = dims.nu = 0;
        const auto np = dims.np = args.np;
        const auto ny = dims.ny = args.ny;
        const auto nz = dims.nz = args.nz;

        // Allocate auxiliary memory
        S.resize(ny, nx + np);
        Hw.resize(nx, nx + np);
        Vw.resize(np, nx + np);
        Jw.resize(nz, nx + np);
        gw.resize(nx);
        xw.resize(nx);
        pw.resize(np);
        yw.resize(ny);
        zw.resize(nz);
        axw.resize(nx);
        apw.resize(np);
        ayw.resize(ny);
        azw.resize(nz);
        bw.resize(ny);
        weights.resize(nx);

        // Initialize the order of x variables
        jx = canonicalizer.Q();
    }

    /// Decompose the canonical saddle point matrix.
    auto decomposeCanonical(CanonicalSaddlePointMatrix args) -> void
    {
        switch(options.method)
        {
        case SaddlePointMethod::Nullspace: nullspace.decompose(args); break;
        case SaddlePointMethod::Rangespace: rangespace.decompose(args); break;
        default: fullspace.decompose(args); break;
        }
    }

    /// Solve the canonical saddle point problem.
    auto solveCanonical(CanonicalSaddlePointProblem args) -> void
    {
        switch(options.method)
        {
        case SaddlePointMethod::Nullspace: nullspace.solve(args); break;
        case SaddlePointMethod::Rangespace: rangespace.solve(args); break;
        default: fullspace.solve(args); break;
        }
    }

    /// Canonicalize the saddle point matrix.
    auto canonicalize(SaddlePointSolverCanonicalize1Args args) -> void
    {
        // Unpack the dimension variables
        auto& [nx, ns, nu, ny, nz, np, nb, nn, nl, nbs, nbu, nns, nnu, nbe, nne, nbi, nni] = dims;

        // Update the number of variables in xu and xs, where x = (xs, xu)
        nu = args.ju.size();
        ns = nx - nu;

        // Ensure number of variables xs is positive.
        assert(ns > 0);

        // Determine if the saddle point matrix is degenerate
        degenerate = ns == 0; // is there one or more variable in xs?

        // Skip the rest if there is no stable variables
        if(degenerate)
            return;

        // Change the ordering of the variables as x = (xs, xu)
        const auto pos = moveIntersectionRight(jx, args.ju);

        // Ensure the indices of xu variables are valid.
        assert(pos == ns && "Cannot proceed with SaddlePointSolver::canonicalize. "
            "There are invalid indices of xu variables.");

        // The indices of the xs and xu variables using jx
        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        // The priority weights of xs and xu variables to become basic variables
        auto ws = weights(js);
        auto wu = weights(ju);

        // Update the priority weights for the update of the canonical form
        ws = abs(args.wx(js));

        // Set the priority weights for the unstable variables xu to negative values.
        // This is done to prevent (as much as possible) the unstable variables
        // in *xu* from becoming basic variables if there are stable variables
        // in *xs* that can be basic instead.
        wu = -linspace(nu, 1, nu);

        // Update the canonical form of Ax
        canonicalizer.updateWithPriorityWeights(weights);

        // Remove residual round-off errors in Sbn after canonical form update.
        // This is an important step to ensure that residual coefficients such
        // as 1.23456e-16 are not present in matrices R and Sbn of the new
        // canonical form R*Ax*Q = [Ibx Sbn].
        canonicalizer.cleanResidualRoundoffErrors();

        // Get the updated indices of basic and non-basic variables
        const auto ibasic = canonicalizer.indicesBasicVariables();
        const auto inonbasic = canonicalizer.indicesNonBasicVariables();

        // Update the number of basic and non-basic variables
        nb = canonicalizer.numBasicVariables();
        nn = canonicalizer.numNonBasicVariables();

        //=========================================================================================
        // Identify stable and unstable basic variables
        //=========================================================================================

        // Update the number of linearly dependent rows in A = [Ax Ap]
        nl = ny - nb;

        // Find the number of basic variables in xu (those with weights below zero)
        // Walk from last to first basic variable, since the unstable basic variables are at the end of ibasic
        nbu = 0;
        for(auto i = 1; i <= nb; ++i)
            if(weights[ibasic[nb - i]] < 0.0) ++nbu;
            else break;

        // Find the number of non-basic variables in xu (those with weights below zero)
        // Walk from last to first non-basic variable, since the unstable non-basic variables are at the end of inonbasic
        nnu = 0;
        for(auto i = 1; i <= nn; ++i)
            if(weights[inonbasic[nn - i]] < 0.0) ++nnu;
            else break;

        // Update the number of stable basic and stable non-basic variables
        nbs = nb - nbu;
        nns = nn - nnu;

        // error(np > 0 && nbs != ny, "Cannot proceed with method "
        //     "SaddlePointSolver::canonicalize. This is due to a current "
        //     "limitation in which if the number of parameter variables *p* "
        //     "(np = " + std::to_string(np) + ") is greater than one, then the "
        //     "coefficient matrix Ax needs to be full rank and equal to the number "
        //     "of basic stable variables (rank[Ax] = " + std::to_string(nb) + ", "
        //     "rows[Ax] = " + std::to_string(ny) + ", nbs = " + std::to_string(nbs) + ").");

        //=========================================================================================
        // Update the order of the stable-variables as xs = (xbe, xbi, xbu, xne, xni, xnu), where:
        // -- xbe are stable explicit basic variables (pivots);
        // -- xbi are stable implicit basic variables (non-pivots);
        // -- xbu are unstable basic variables;
        // -- xne are stable explicit non-basic variables (pivots);
        // -- xni are stable implicit non-basic variables (non-pivots);
        // -- xnu are unstable non-basic variables.
        //-----------------------------------------------------------------------------------------
        // Note: Pivot and non-pivot as in the Gaussian elimination sense. The pivot variables
        // are those whose the corresponding diagonal entry in the H matrix is not
        // dominant in the infinity norm.
        //=========================================================================================

        // Initialize the permutation matrices Kb and Kn with identity state
        Kb = indices(nb);
        Kn = indices(nn);

        // The sub-vectors Kbs and Kns in Kb = [Kbs, Kbu] and Kn = [Kns, Knu, Knp]
        // Note: Keep Kbu intact below to preserve position of xu basic
        // variables (already at the end!). The same applies for Knu and Knp,
        // since xnp variables are already at the very end (because of -inf
        // priority weights) and the xnu variables before xnp variables and
        // after xns variables.
        auto Kbs = Kb.head(nbs);
        auto Kns = Kn.head(nns);

        // The diagonal entries in the Hxx matrix.
        const auto Hd = args.Hxx.diagonal();

        // Sort the basic and non-basic stable-variables in decreasing order w.r.t. absolute values of H diagonal entries.
        std::sort(Kbs.begin(), Kbs.end(), [&](auto l, auto r) { return abs(Hd[ibasic[l]]) > abs(Hd[ibasic[r]]); }); // Note: Use > for stricter comparison, not >=!
        std::sort(Kns.begin(), Kns.end(), [&](auto l, auto r) { return abs(Hd[inonbasic[l]]) > abs(Hd[inonbasic[r]]); }); // Note: Use > for stricter comparison, not >=!

        // Update the ordering of the basic and non-basic variables in the canonicalizer object
        canonicalizer.updateOrdering(Kb, Kn);

        // Get the newly ordered indices of basic and non-basic variables after the above ordering update
        const auto jb = canonicalizer.indicesBasicVariables();
        const auto jn = canonicalizer.indicesNonBasicVariables();

        // View to the sub-matrices Sbn and Sbp in S = [Sbn Sbp] = [Sbsns Sbsnu Sbsp; 0 Sbunu Sbup]
        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        // The top nb rows of R = [Rt; 0l] where 0l is present in case of rank deficiency of Ax.
        const auto Rt = canonicalizer.R().topRows(nb);

        // Update Sbn from the just canonicalized matrx Wx = [Ax; Jx]
        Sbn = canonicalizer.S();

        // Compute the Sbp matrix to complete the canonicalization of A = [Ax Ap], i.e. R*A*Q' = [Ibb Sbn Sbp]
        Sbp = Rt * Ap;

        // Remove also residual round-off errors from Sbp.
        cleanResidualRoundoffErrors(Sbp);

        // Return true if the i-th basic variable is a pivot/explicit variable
        const auto is_basic_explicit = [&](auto i)
        {
            const auto idx = jb[i];                     // the global index of the basic variable
            const auto Hii = Hd[idx];                   // the corresponding diagonal entry in the H matrix
            const auto a1 = 1.0;                        // the max value along the corresponding column of the identity matrix
            const auto a2 = norminf(args.Vpx.col(idx)); // the max value along the corresponding column of the Vpx matrix
            return abs(Hii) >= std::max(a1, a2);        // return true if diagonal entry is dominant with respect to Vpx and Ibb only (not Hxx!)
        };

        // Return true if the i-th non-basic variable is a pivot/explicit variable
        const auto is_nonbasic_explicit = [&](auto i)
        {
            const auto idx = jn[i];                     // the global index of the non-basic variable
            const auto Hii = Hd[idx];                   // the corresponding diagonal entry in the H matrix
            const auto a1 = norminf(Sbn.col(i));        // the max value along the corresponding column of the Sbn matrix
            const auto a2 = norminf(args.Vpx.col(idx)); // the max value along the corresponding column of the Vpx matrix
            return abs(Hii) >= std::max(a1, a2);        // return true if diagonal entry is dominant with respect to Vpx and Sbn only (not Hxx!)
        };

        // Find the number of pivot basic variables (those with |Hbebe| >= I and |Hbebe| >= |cols(H, jbe)|)
        // Walk from first to last stable basic variable, since they are ordered in decresiang order of |Hii| values
        nbe = 0; while(nbe < nbs && is_basic_explicit(nbe)) ++nbe;

        // Find the number of pivot non-basic variables (those with |Hnene| >= |Sbsne| and |Hnene| > |cols(H, jne)|)
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

        // Update the order of x variables as x = (xs, xu) = (xbs, xns, xbu, xnu)
        jx << jbs, jns, jbu, jnu;

        //=========================================================================================
        // Initialize matrices Hss, Hsp, Vps, Vpu, Vpp, Js, Ju, Jp
        //=========================================================================================

        // Views to the sub-matrices Hss, Hsp
        auto Hss = Hw.topLeftCorner(ns, ns);
        auto Hsp = Hw.topRightCorner(ns, np);

        // Views to the sub-matrices in J = [Jx Jp] = [Js Ju Jp]
        auto Jx = Jw.leftCols(nx);
        auto Jp = Jw.rightCols(np);

        // Views to the sub-matrices in V = [Vpx Vpp] = [Vps Vpu Vpp]
        auto Vpx = Vw.leftCols(nx);
        auto Vpp = Vw.rightCols(np);

        // Initialize matrices Hss and Hsp taking into account the method in use
        Hsp = rows(args.Hxp, js);
        if(options.method == SaddlePointMethod::Rangespace)
             Hss.diagonal() = args.Hxx.diagonal()(js);
        else Hss = args.Hxx(js, js);

        // Initialize matrices Vpx and Vpp
        Vpx = cols(args.Vpx, jx);
        Vpp = args.Vpp;

        // Initialize matrices Jx and Jp
        Jx = cols(args.Jx, jx);
        Jp = args.Jp;
    }

    /// Compute the right-hand side vector in the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto rhs(SaddlePointSolverRhs1Args args) -> void
    {
        const auto nx  = dims.nx;
        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;
        const auto nl  = dims.nl;

        const auto R  = canonicalizer.R();

        axw = args.ax(jx);
        apw = args.ap;
        azw = args.az;

        // NOTE: It is extremely important to use this logic below, of
        // eliminating contribution in ay from unstable-variables using matrix
        // Au instead of working on the canonical level, using matrices Sbsnu
        // and Sbunu. By doing this, we can better control the feasibility
        // error when the unstable-variables correspond to variables on lower
        // bounds (i.e. 1e-40) and they can contaminate the canonical
        // residuals. If they are very small, they will either vanish in the
        // operation below using R or via the clean residual round-off errors.
        ayw.noalias() = R * args.ay;

        // Ensure residual round-off errors are cleaned in ay'.
        // This improves accuracy and stability in degenerate cases when some
        // variables need to have tiny values. For example, suppose the
        // equality constraints effectively enforce `xi - 2xj = 0`, for
        // variables `i` and `j`, which, combined with the optimality
        // constraints, results in an exact solution of `xj = 1e-31`. If,
        // instead, we don't clean these residual round-off errors, we may
        // instead have `xi - 2xj = eps`, where `eps` is a small residual error
        // (e.g., 1e-16). This will cause `xi = eps`, and not `xi = 2e-31`.
        cleanResidualRoundoffErrors(ayw);

        auto aybu = ayw.segment(nbs, nbu);
        auto ayl  = ayw.tail(nl);

        aybu.fill(0.0); // ensure residuals w.r.t. basic unstable variables are zero
        ayl.fill(0.0); // ensure residuals w.r.t. linearly dependent rows in Ax are zero
    }

    /// Compute the right-hand side vector in the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto rhs(SaddlePointSolverRhs2Args args) -> void
    {
        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto np  = dims.np;
        const auto ny  = dims.ny;
        const auto nz  = dims.nz;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;
        const auto nl  = dims.nl;

        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        const auto As = cols(Ax, js);
        const auto Au = cols(Ax, ju);

        const auto Js = Jw.leftCols(ns);
        const auto Jp = Jw.rightCols(np);

        const auto R = canonicalizer.R();

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        const auto xs = xw.head(ns) = args.x(js);
        const auto xu = xw.tail(nu) = args.x(ju);

        const auto xbs = xs.head(nbs);
        const auto xns = xs.tail(nns);

        const auto gs = gw.head(ns) = args.fx(js);

        const auto bbs = bw.head(nbs);

        auto as   = axw.head(ns);
        auto au   = axw.tail(nu);
        auto ap   = apw.head(np);
        auto az   = azw.head(nz);
        auto aybs = ayw.head(nbs);
        auto aybu = ayw.segment(nbs, nbu);
        auto ayl  = ayw.tail(nl);

        as.noalias() = -(gs + tr(As) * args.y + tr(Js) * args.z);
        au.fill(0.0);

        ap = -args.v;
        az = -args.h;

        bw.noalias() = R * (args.b - Au * xu); // eliminate contribution from unstable variables and apply R!

        cleanResidualRoundoffErrors(bw); // ensure residual round-off errors are removed! For example, removing 1e-15 among numbers 1.2, 55.2

        aybs = bbs - xbs - Sbsns * xns - Sbsp * args.p; // TODO: This produces round-off errors. Maybe consider the other rhs overload that does not need this computation.
        aybu.fill(0.0); // ensure residuals w.r.t. basic unstable variables are zero
        ayl.fill(0.0); // ensure residuals w.r.t. linearly dependent rows in Ax are zero
    }

    /// Compute the right-hand side vector in the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto rhs(SaddlePointSolverRhs3Args args) -> void
    {
        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto np  = dims.np;
        const auto ny  = dims.ny;
        const auto nz  = dims.nz;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;
        const auto nl  = dims.nl;

        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        const auto As = cols(Ax, js);
        const auto Au = cols(Ax, ju);

        const auto Hss = Hw.topLeftCorner(ns, ns);
        const auto Hsp = Hw.topRightCorner(ns, np);

        const auto Js = Jw.leftCols(ns);
        const auto Jp = Jw.rightCols(np);

        const auto Vps = Vw.leftCols(ns);
        const auto Vpp = Vw.rightCols(np);

        const auto R = canonicalizer.R();

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        const auto xs = xw.head(ns) = args.x(js);
        const auto xu = xw.tail(nu) = args.x(ju);

        const auto xbs = xs.head(nbs);
        const auto xns = xs.tail(nns);

        const auto gs = gw.head(ns) = args.fx(js);

        const auto bbs = bw.head(nbs);

        auto as   = axw.head(ns);
        auto au   = axw.tail(nu);
        auto ap   = apw.head(np);
        auto az   = azw.head(nz);
        auto aybs = ayw.head(nbs);
        auto aybu = ayw.segment(nbs, nbu);
        auto ayl  = ayw.tail(nl);

        bw.noalias() = R * (args.b - Au * xu); // eliminate contribution from unstable variables and apply R!

        cleanResidualRoundoffErrors(bw); // ensure residual round-off errors are removed! For example, removing 1e-15 among numbers 1.2, 55.2

        as = Hsp*args.p - gs;
        if(options.method == SaddlePointMethod::Rangespace)
            as += Hss.diagonal().cwiseProduct(xs);
        else as.noalias() += Hss*xs;

        au = xu;

        ap = Vps*xs + Vpp*args.p - args.v;
        az = Js*xs + Jp*args.p - args.h;

        aybs = bbs;
        aybu.fill(0.0); // ensure residuals w.r.t. basic unstable variables are zero
        ayl.fill(0.0); // ensure residuals w.r.t. linearly dependent rows in Ax are zero
    }

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto decompose() -> void
    {
        const auto ns  = dims.ns;
        const auto np  = dims.np;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;

        const auto Hss = Hw.topLeftCorner(ns, ns);
        const auto Hsp = Hw.topRightCorner(ns, np);

        const auto Js = Jw.leftCols(ns);
        const auto Jp = Jw.rightCols(np);

        const auto Vps = Vw.leftCols(ns);
        const auto Vpp = Vw.rightCols(np);

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        decomposeCanonical({ dims, Hss, Hsp, Vps, Vpp, Js, Jp, Sbsns, Sbsp });
    }

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose has been called before this method.
    auto solve(SaddlePointSolverSolve1Args args) -> void
    {
        const auto nx  = dims.nx;
        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto np  = dims.np;
        const auto nz  = dims.nz;
        const auto ny  = dims.ny;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;
        const auto nl  = dims.nl;

        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        const auto Hss = Hw.topLeftCorner(ns, ns);
        const auto Hsp = Hw.topRightCorner(ns, np);

        const auto Js = Jw.leftCols(ns);
        const auto Jp = Jw.rightCols(np);

        const auto Vps = Vw.leftCols(ns);
        const auto Vpp = Vw.rightCols(np);

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        const auto R = canonicalizer.R();

        const auto as   = axw.head(ns);
        const auto au   = axw.tail(nu);
        const auto ap   = apw.head(np);
        const auto az   = azw.head(nz);
        const auto aybs = ayw.head(nbs);

        auto x   = xw.head(nx);
        auto xs  = xw.head(ns);
        auto xu  = xw.tail(nu);
        auto y   = yw.head(ny);
        auto ybs = y.head(nbs);
        auto ybu = y.segment(nbs, nbu);
        auto yl  = y.tail(nl);
        auto p   = args.sp;
        auto z   = args.sz;

        ybu.fill(0.0); // zero for y values associated with unstable basic variables
        yl.fill(0.0);  // zero for y values associated with linearly dependent rows in Ax

        solveCanonical({ dims, Hss, Hsp, Vps, Vpp, Js, Jp, Sbsns, Sbsp, as, ap, az, aybs, xs, p, z, ybs });

        args.sx(js) = xs;
        args.sx(ju) = au;
        args.sp = p;
        args.sz = z;
        args.sy.noalias() = tr(R) * y;
    }

    /// Multiply the saddle point matrix with a given vector.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto multiply(SaddlePointSolverMultiplyArgs args) -> void
    {
        const auto nx = dims.nx;
        const auto ns = dims.ns;
        const auto nu = dims.nu;
        const auto np = dims.np;

        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        const auto Hss = Hw.topLeftCorner(ns, ns);
        const auto Hsp = Hw.topRightCorner(ns, np);

        const auto Jx = Jw.leftCols(nx);
        const auto Jp = Jw.rightCols(np);
        const auto Js = Jx.leftCols(ns);

        const auto Vpx = Vw.leftCols(nx);
        const auto Vpp = Vw.rightCols(np);
        const auto Vps = Vpx.leftCols(ns);

        const auto As = cols(Ax, js);

        auto as = args.ax(js);
        auto au = args.ax(ju);
        auto ap = args.ap;
        auto ay = args.ay;
        auto az = args.az;

        const auto rs = xw.head(ns) = args.rx(js);
        const auto ru = xw.tail(nu) = args.rx(ju);
        const auto rp = args.rp;
        const auto ry = args.ry;
        const auto rz = args.rz;

        as = Hsp*rp + tr(As)*ry + tr(Js)*rz;
        if(options.method == SaddlePointMethod::Rangespace)
            as += Hss.diagonal().cwiseProduct(rs);
        else as += Hss*rs;

        au.noalias() = ru;
        ap.noalias() = Vps*rs + Vpp*rp;
        az.noalias() = Js*rs + Jp*rp;
        ay.noalias() = As*rs + Ap*rp;
    }

    /// Multiply the transpose of the saddle point matrix with a given vector.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto transposeMultiply(SaddlePointSolverTransposeMultiplyArgs args) -> void
    {
        const auto nx = dims.nx;
        const auto ns = dims.ns;
        const auto nu = dims.nu;
        const auto np = dims.np;

        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        const auto Hss = Hw.topLeftCorner(ns, ns);
        const auto Hsp = Hw.topRightCorner(ns, np);

        const auto Jx = Jw.leftCols(nx);
        const auto Jp = Jw.rightCols(np);
        const auto Js = Jx.leftCols(ns);

        const auto Vpx = Vw.leftCols(nx);
        const auto Vpp = Vw.rightCols(np);
        const auto Vps = Vpx.leftCols(ns);

        const auto As = cols(Ax, js);

        auto as = args.ax(js);
        auto au = args.ax(ju);
        auto ap = args.ap;
        auto ay = args.ay;
        auto az = args.az;

        const auto rs = xw.head(ns) = args.rx(js);
        const auto ru = xw.tail(nu) = args.rx(ju);
        const auto rp = args.rp;
        const auto ry = args.ry;
        const auto rz = args.rz;

        as = tr(Vps)*rp + tr(Js)*rz + tr(As)*ry;
        if(options.method == SaddlePointMethod::Rangespace)
            as += Hss.diagonal().cwiseProduct(rs);
        else as += tr(Hss)*rs;

        au = ru;
        ap.noalias() = tr(Hsp)*rs + tr(Vpp)*rp + tr(Jp)*rz + tr(Ap)*ry;
        az.noalias() = Js*rs;
        ay.noalias() = As*rs;
    }

    /// Return the state of the canonical saddle point solver.
    auto state() const -> SaddlePointSolverState
    {
        const auto nx  = dims.nx;
        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto np  = dims.np;
        const auto nz  = dims.nz;
        const auto ny  = dims.ny;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;
        const auto nnu = dims.nnu;
        const auto nl  = dims.nl;

        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        const auto jbs = js.head(nbs);
        const auto jns = js.tail(nns);

        const auto jbu = ju.head(nbu);
        const auto jnu = ju.tail(nnu);

        const auto Hss = Hw.topLeftCorner(ns, ns);
        const auto Hsp = Hw.topRightCorner(ns, np);

        const auto Js = Jw.leftCols(ns);
        const auto Jp = Jw.rightCols(np);

        const auto As = cols(Ax, js);
        const auto Au = cols(Ax, ju);

        const auto Vps = Vw.leftCols(ns);
        const auto Vpp = Vw.rightCols(np);

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        const auto R = canonicalizer.R();

        const auto as = axw.head(ns);
        const auto au = axw.tail(nu);
        const auto ap = apw.head(np);
        const auto az = azw.head(nz);
        const auto ay = ayw.head(ny);

        return { dims, js, jbs, jns, ju, jbu, jnu, R, Hss, Hsp, Vps, Vpp,
            As, Au, Ap, Js, Jp, Sbsns, Sbsp, as, au, ap, ay, az };
    }
};

SaddlePointSolver::SaddlePointSolver(SaddlePointSolverInitArgs args)
: pimpl(new Impl(args))
{}

SaddlePointSolver::SaddlePointSolver(const SaddlePointSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

SaddlePointSolver::~SaddlePointSolver()
{}

auto SaddlePointSolver::operator=(SaddlePointSolver other) -> SaddlePointSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto SaddlePointSolver::setOptions(const SaddlePointOptions& options) -> void
{
    pimpl->options = options;
}

auto SaddlePointSolver::options() const -> const SaddlePointOptions&
{
    return pimpl->options;
}

auto SaddlePointSolver::canonicalize(SaddlePointSolverCanonicalize1Args args) -> void
{
    pimpl->canonicalize(args);
}

auto SaddlePointSolver::rhs(SaddlePointSolverRhs1Args args) -> void
{
    pimpl->rhs(args);
}

auto SaddlePointSolver::rhs(SaddlePointSolverRhs2Args args) -> void
{
    pimpl->rhs(args);
}

auto SaddlePointSolver::rhs(SaddlePointSolverRhs3Args args) -> void
{
    pimpl->rhs(args);
}

auto SaddlePointSolver::decompose() -> void
{
    pimpl->decompose();
}

auto SaddlePointSolver::solve(SaddlePointSolverSolve1Args args) -> void
{
    pimpl->solve(args);
}

auto SaddlePointSolver::multiply(SaddlePointSolverMultiplyArgs args) -> void
{
    pimpl->multiply(args);
}

auto SaddlePointSolver::transposeMultiply(SaddlePointSolverTransposeMultiplyArgs args) -> void
{
    pimpl->transposeMultiply(args);
}

auto SaddlePointSolver::state() const -> SaddlePointSolverState
{
    return pimpl->state();
}

} // namespace Optima
