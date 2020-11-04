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
#include <Optima/Exception.hpp>
#include <Optima/EchelonizerExtended.hpp>
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
    EchelonizerExtended echelonizer;    ///< The echelonizer of matrix *Wx = [Ax; Jx]*.
    SaddlePointSolverRangespace rangespace; ///< The canonical saddle point solver based on a rangespace algorithm.
    SaddlePointSolverNullspace nullspace;   ///< The canonical saddle point solver based on a nullspace algorithm.
    SaddlePointSolverFullspace fullspace;   ///< The canonical saddle point solver based on a fullspace algorithm.
    SaddlePointOptions options;             ///< The options used to solve the saddle point problems.
    SaddlePointDims dims;                   ///< The dimensions of the saddle point problem.
    const Matrix Ax;                        ///< The constant matrix Ax of the saddle point problem.
    const Matrix Ap;                        ///< The constant matrix Ap of the saddle point problem.
    Matrix W;                               ///< The matrix W = [Wx Wp] = [Ws Wu Wp] = [As Au Ap; Js Ju Jp] of the saddle point problem.
    Matrix S;                               ///< The S = [Sbsns Sbsnu Sbsp; 0 Sbunu Sbup] matrix that stores the canonical form of W = [Ax Ap; Jx Jp].
    Matrix H;                               ///< The workspace for matrix H.
    Matrix V;                               ///< The workspace for matrix V.
    Vector g;                               ///< The workspace for right-hand side vector g
    Vector asu;                             ///< The workspace for right-hand side vector asu = (as, au)
    Vector ap;                              ///< The workspace for right-hand side vector ap
    Vector ay;                              ///< The workspace for right-hand side vector ay
    Vector az;                              ///< The workspace for right-hand side vector az
    Vector aw;                              ///< The workspace for right-hand side vector aw = (ay, az)
    Vector awstar;                          ///< The workspace for auxiliary vector aw(star)
    Vector xsu;                             ///< The workspace for vector xsu = (xs, xu)
    Vector w;                               ///< The workspace for solution vector w = (y, z)
    Vector waux;                            ///< The workspace for auxiliary vector waux = tr(R) * w
    Vector weights;                         ///< The priority weights for the selection of basic variables.
    Indices jx;                             ///< The order of x variables as x = (xbs, xns, xbu, xnu).
    Indices Kb;                             ///< The permutation matrix used to order the basic variables as xb = (xbe, xbi, xbu) with `e` and `i` denoting pivot and non-pivot
    Indices Kn;                             ///< The permutation matrix used to order the non-basic variables as xn = (xne, xni, xnu) with `e` and `i` denoting pivot and non-pivot

    /// Construct a SaddlePointSolver::Impl instance with given data.
    Impl(SaddlePointSolverInitArgs args)
    : echelonizer(args.Ax),
      rangespace(args.nx, args.np, args.ny + args.nz),
      nullspace(args.nx, args.np, args.ny + args.nz),
      fullspace(args.nx, args.np, args.ny + args.nz),
      Ax(args.Ax), Ap(args.Ap)
    {
        // Ensure consistent and proper dimensions
        assert(args.nx > 0);
        assert(args.Ax.rows() == 0 || args.Ax.rows() == args.ny);
        assert(args.Ap.rows() == 0 || args.Ap.rows() == args.ny);
        assert(args.Ax.rows() == 0 || args.Ax.cols() == args.nx);
        assert(args.Ap.rows() == 0 || args.Ap.cols() == args.np);

        // Initialize the number of variables x, xs, xu, p, y, z, w = (y, z)
        const auto nx = dims.nx = args.nx;
        const auto ns = dims.ns = nx;
        const auto nu = dims.nu = 0;
        const auto np = dims.np = args.np;
        const auto ny = dims.ny = args.ny;
        const auto nz = dims.nz = args.nz;
        const auto nw = dims.nw = ny + nz;

        // Allocate auxiliary memory
        W.resize(nw, nx + np);
        S.resize(nw, nx + np);
        H.resize(nx, nx + np);
        V.resize(np, nx + np);
        g.resize(nx);
        xsu.resize(nx);
        w.resize(nw);
        waux.resize(nw);
        asu.resize(nx);
        ap.resize(np);
        ay.resize(ny);
        az.resize(nz);
        aw.resize(nw);
        awstar.resize(nw);
        weights.resize(nx);

        // Initialize the order of x variables
        jx = echelonizer.Q();
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
        auto& [nx, ns, nu, ny, nz, nw, np, nb, nn, nl, nbs, nbu, nns, nnu, nbe, nne, nbi, nni] = dims;

        // Update the number of variables in xu and xs, where x = (xs, xu)
        nu = args.ju.size();
        ns = nx - nu;

        // Ensure number of variables xs is positive.
        assert(ns > 0);

        // Change the ordering of the variables as x = (xs, xu)
        const auto pos = moveIntersectionRight(jx, args.ju);

        // Ensure the indices of xu variables are valid.
        assert(pos == ns && "Cannot proceed with SaddlePointSolver::canonicalize. "
            "There are out-of-range indices of unstable variables.");

        // The indices of the xs and xu variables using jx
        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        // Update the priority weights for the update of the canonical form (determination of the basic variables)
        weights(js) = abs(args.wx(js));

        // Set the priority weights for the unstable variables to negative
        // values. This is done to prevent (as much as possible) the unstable
        // variables in *x* from becoming basic variables if there are stable
        // variables that can be basic instead.
        weights(ju) = -linspace(nu, 1, nu);

        // Update the canonical form of Wx = [Ax; Jx]
        echelonizer.updateWithPriorityWeights(args.Jx, weights);

        // Remove residual round-off errors in Sbn after canonical form update.
        // This is an important step to ensure that residual coefficients such
        // as 1.23456e-16 are not present in matrices R and Sbn of the new
        // canonical form R*Ax*Q = [Ibx Sbn].
        echelonizer.cleanResidualRoundoffErrors();

        // Get the updated indices of basic and non-basic variables
        const auto ibasic = echelonizer.indicesBasicVariables();
        const auto inonbasic = echelonizer.indicesNonBasicVariables();

        // Update the number of basic and non-basic variables
        nb = echelonizer.numBasicVariables();
        nn = echelonizer.numNonBasicVariables();

        //=========================================================================================
        // Identify stable and unstable basic variables
        //=========================================================================================

        // Update the number of linearly dependent rows in Wx = [Ax; Jx]
        nl = nw - nb;

        // Find the number of basic variables (those with weights below zero)
        // Walk from last to first basic variable, since the unstable basic variables are at the end of ibasic
        nbu = 0;
        for(auto i = 1; i <= nb; ++i)
            if(weights[ibasic[nb - i]] < 0.0) ++nbu;
            else break;

        // Find the number of non-basic variables (those with weights below zero)
        // Walk from last to first non-basic variable, since the unstable non-basic variables are at the end of inonbasic
        nnu = 0;
        for(auto i = 1; i <= nn; ++i)
            if(weights[inonbasic[nn - i]] < 0.0) ++nnu;
            else break;

        // Update the number of stable basic and stable non-basic variables
        nbs = nb - nbu;
        nns = nn - nnu;

        // error(np > 0 && nbs != nw, "Cannot proceed with method "
        //     "SaddlePointSolver::canonicalize. This is due to a current "
        //     "limitation in which if the number of parameter variables *p* "
        //     "(np = " + std::to_string(np) + ") is greater than one, then the "
        //     "coefficient matrix Wx = [Ax; Jx] needs to be full rank and equal to the number "
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
        std::sort(Kbs.begin(), Kbs.end(), [&](auto l, auto r) { return abs(Hd[ibasic[l]]) > abs(Hd[ibasic[r]]); }); // Note: Use of > for stricter comparison during sort is needed. Don't use >=!
        std::sort(Kns.begin(), Kns.end(), [&](auto l, auto r) { return abs(Hd[inonbasic[l]]) > abs(Hd[inonbasic[r]]); }); // Note: Use of > for stricter comparison during sort is needed. Don't use >=!

        // Update the ordering of the basic and non-basic variables in the echelonizer object
        echelonizer.updateOrdering(Kb, Kn);

        // Get the newly ordered indices of basic and non-basic variables after the above ordering update
        const auto jb = echelonizer.indicesBasicVariables();
        const auto jn = echelonizer.indicesNonBasicVariables();

        //=========================================================================================
        // Initialize matrix block Wp in W = [Ws Wu Wp]
        //=========================================================================================

        // Update the Wp block in W = [Ws Wu Wp] = [As Au Ap; Js Ju Jp]
        auto Wp = W.rightCols(np);

        Wp.topRows(ny) = Ap;
        Wp.bottomRows(nz) = args.Jp;

        //=========================================================================================
        // Initialize matrix S = [Sbn Sbp] = [Sbsns Sbsnu Sbsp; 0 Sbunu Sbup]
        //=========================================================================================

        // The top nb rows of R = [Rb; 0l] where 0l is present in case of rank deficiency of Wx = [Ax; Jx].
        const auto Rb = echelonizer.R().topRows(nb);

        // View to the sub-matrices Sbn and Sbp in S = [Sbn Sbp]
        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        // Update Sbn from the just canonicalized matrx Wx = [Ax; Jx]
        Sbn = echelonizer.S();

        // Compute Sbp = Rb*Wp
        Sbp = Rb * Wp;

        // Remove also residual round-off errors from Sbp.
        cleanResidualRoundoffErrors(Sbp);

        //=========================================================================================
        // Identify the explicit/implicit basic/non-basic variables
        //=========================================================================================

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

        // Update the order of x variables as x = (xs, xu) = (xbs, xns, xbu, xnu)
        jx << jbs, jns, jbu, jnu;

        //=========================================================================================
        // Initialize matrices Hss, Hsp
        //=========================================================================================

        // Views to the sub-matrices Hss, Hsp
        auto Hss = H.topLeftCorner(ns, ns);
        auto Hsp = H.topRightCorner(ns, np);

        // Initialize matrices Hss and Hsp taking into account the method in use
        Hsp = rows(args.Hxp, js);
        if(options.method == SaddlePointMethod::Rangespace)
             Hss.diagonal() = args.Hxx.diagonal()(js);
        else Hss = args.Hxx(js, js);

        //=========================================================================================
        // Initialize matrices Vps, Vpu, Vpp
        //=========================================================================================

        // Views to the sub-matrices in V = [Vpx Vpp] = [Vps Vpu Vpp]
        auto Vpx = V.leftCols(nx);
        auto Vpp = V.rightCols(np);

        // Initialize matrices Vpx and Vpp
        Vpx = cols(args.Vpx, jx);
        Vpp = args.Vpp;

        //=========================================================================================
        // Initialize matrices Ws = [As; Js] and Wu = [Au; Ju] in W = [As Au Ap; Js Ju Jp]
        //=========================================================================================

        // Views to the sub-matrices in W = [Ws Wu Wp] = [Wsu Wp] = [Asu Ap; Jsu Jp]
        auto Wsu = W.leftCols(nx);
        auto Asu = Wsu.topRows(ny);
        auto Jsu = Wsu.bottomRows(nz);

        // Initialize matrices Ws and Wu.
        Asu = cols(Ax, jx);
        Jsu = cols(args.Jx, jx);
    }

    /// Canonicalize the saddle point matrix.
    auto canonicalize(SaddlePointSolverCanonicalize2Args args) -> void
    {
        // Update the priority weights to determine the basic variables in the
        // canonical form Note that those non-positive priority weights are
        // replaced with -1.0. This is to give a fair chance for the
        // potentially unstable variables (attached to their bounds, and thus
        // with zero priority weight) to become basic variables in case there
        // are no other stable variable that could be basic.
        weights = (args.wx.array() > 0.0).select(args.wx, -1.0);

        // Update the canonical form of Wx = [Ax; Jx]
        echelonizer.updateWithPriorityWeights(args.Jx, weights);

        // Remove residual round-off errors in Sbn after canonical form update.
        // This is an important step to ensure that residual coefficients such
        // as 1.23456e-16 are not present in matrices R and Sbn of the new
        // canonical form R*Ax*Q = [Ibx Sbn].
        echelonizer.cleanResidualRoundoffErrors();
    }

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto decompose() -> void
    {
        const auto ns  = dims.ns;
        const auto np  = dims.np;
        const auto nbs = dims.nbs;
        const auto nns = dims.nns;

        const auto Hss = H.topLeftCorner(ns, ns);
        const auto Hsp = H.topRightCorner(ns, np);

        const auto Vps = V.leftCols(ns);
        const auto Vpp = V.rightCols(np);

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        decomposeCanonical({ dims, Hss, Hsp, Vps, Vpp, Sbsns, Sbsp });
    }

    /// Decompose the coefficient matrix of the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto decompose(SaddlePointSolverDecomposeArgs args) -> void
    {
        // Unpack the dimension variables
        auto& [nx, ns, nu, ny, nz, nw, np, nb, nn, nl, nbs, nbu, nns, nnu, nbe, nne, nbi, nni] = dims;

        // Update the number of variables in xu and xs, where x = (xs, xu)
        nu = args.ju.size();
        ns = nx - nu;

        // Ensure number of variables xs is positive.
        assert(ns > 0);

        // Change the ordering of the variables as x = (xs, xu)
        const auto pos = moveIntersectionRight(jx, args.ju);

        // Ensure the indices of xu variables are valid.
        assert(pos == ns && "Cannot proceed with SaddlePointSolver::canonicalize. "
            "There are out-of-range indices of unstable variables.");

        // The indices of the xs and xu variables using jx
        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        // Get the updated indices of basic and non-basic variables
        const auto ibasic = echelonizer.indicesBasicVariables();
        const auto inonbasic = echelonizer.indicesNonBasicVariables();

        // Update the number of basic and non-basic variables
        nb = echelonizer.numBasicVariables();
        nn = echelonizer.numNonBasicVariables();

        //=========================================================================================
        // Identify stable and unstable basic variables
        //=========================================================================================

        // Update the number of linearly dependent rows in Wx = [Ax; Jx]
        nl = nw - nb;

        weights(js).fill(+1.0);
        weights(ju).fill(-1.0);

        // Find the number of basic variables (those with weights below zero)
        // Walk from last to first basic variable, since the unstable basic variables are at the end of ibasic
        nbu = 0;
        for(auto i = 1; i <= nb; ++i)
            if(weights[ibasic[nb - i]] < 0.0)
                ++nbu;

        // Find the number of non-basic variables (those with weights below zero)
        // Walk from last to first non-basic variable, since the unstable non-basic variables are at the end of inonbasic
        nnu = 0;
        for(auto i = 1; i <= nn; ++i)
            if(weights[inonbasic[nn - i]] < 0.0)
                ++nnu;

        // Update the number of stable basic and stable non-basic variables
        nbs = nb - nbu;
        nns = nn - nnu;

        // error(np > 0 && nbs != nw, "Cannot proceed with method "
        //     "SaddlePointSolver::canonicalize. This is due to a current "
        //     "limitation in which if the number of parameter variables *p* "
        //     "(np = " + std::to_string(np) + ") is greater than one, then the "
        //     "coefficient matrix Wx = [Ax; Jx] needs to be full rank and equal to the number "
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
        std::sort(Kbs.begin(), Kbs.end(), [&](auto l, auto r) { return abs(Hd[ibasic[l]]) > abs(Hd[ibasic[r]]); }); // Note: Use of > for stricter comparison during sort is needed. Don't use >=!
        std::sort(Kns.begin(), Kns.end(), [&](auto l, auto r) { return abs(Hd[inonbasic[l]]) > abs(Hd[inonbasic[r]]); }); // Note: Use of > for stricter comparison during sort is needed. Don't use >=!

        // Update the ordering of the basic and non-basic variables in the echelonizer object
        echelonizer.updateOrdering(Kb, Kn);

        // Get the newly ordered indices of basic and non-basic variables after the above ordering update
        const auto jb = echelonizer.indicesBasicVariables();
        const auto jn = echelonizer.indicesNonBasicVariables();

        //=========================================================================================
        // Initialize matrix block Wp in W = [Ws Wu Wp]
        //=========================================================================================

        // Update the Wp block in W = [Ws Wu Wp] = [As Au Ap; Js Ju Jp]
        auto Wp = W.rightCols(np);

        Wp.topRows(ny) = Ap;
        Wp.bottomRows(nz) = args.Jp;

        //=========================================================================================
        // Initialize matrix S = [Sbn Sbp] = [Sbsns Sbsnu Sbsp; 0 Sbunu Sbup]
        //=========================================================================================

        // The top nb rows of R = [Rb; 0l] where 0l is present in case of rank deficiency of Wx = [Ax; Jx].
        const auto Rb = echelonizer.R().topRows(nb);

        // View to the sub-matrices Sbn and Sbp in S = [Sbn Sbp]
        auto Sbn = S.topLeftCorner(nb, nn);
        auto Sbp = S.topRightCorner(nb, np);

        // View to the sub-matrices Sbsns and Sbsp in S = [Sbsns Sbsnu Sbsp; 0 Sbunu Sbup]
        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        // Update Sbn from the just canonicalized matrx Wx = [Ax; Jx]
        Sbn = echelonizer.S();

        // Compute Sbp = Rb*Wp
        Sbp = Rb * Wp;

        // Remove also residual round-off errors from Sbp.
        cleanResidualRoundoffErrors(Sbp);

        //=========================================================================================
        // Identify the explicit/implicit basic/non-basic variables
        //=========================================================================================

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

        // Update the order of x variables as x = (xs, xu) = (xbs, xns, xbu, xnu)
        jx << jbs, jns, jbu, jnu;

        //=========================================================================================
        // Initialize matrices Hss, Hsp
        //=========================================================================================

        // Views to the sub-matrices Hss, Hsp
        auto Hss = H.topLeftCorner(ns, ns);
        auto Hsp = H.topRightCorner(ns, np);

        // Initialize matrices Hss and Hsp taking into account the method in use
        Hsp = rows(args.Hxp, js);
        if(options.method == SaddlePointMethod::Rangespace)
             Hss.diagonal() = args.Hxx.diagonal()(js);
        else Hss = args.Hxx(js, js);

        //=========================================================================================
        // Initialize matrices Vps, Vpu, Vpp
        //=========================================================================================

        // Views to the sub-matrices in V = [Vpx Vpp] = [Vps Vpu Vpp]
        auto Vpx = V.leftCols(nx);
        auto Vps = V.leftCols(ns);
        auto Vpp = V.rightCols(np);

        // Initialize matrices Vpx and Vpp
        Vpx = cols(args.Vpx, jx);
        Vpp = args.Vpp;

        //=========================================================================================
        // Initialize matrices Ws = [As; Js] and Wu = [Au; Ju] in W = [As Au Ap; Js Ju Jp]
        //=========================================================================================

        // Views to the sub-matrices in W = [Ws Wu Wp] = [Wsu Wp] = [Asu Ap; Jsu Jp]
        auto Wsu = W.leftCols(nx);
        auto Asu = Wsu.topRows(ny);
        auto Jsu = Wsu.bottomRows(nz);

        // Initialize matrices Ws and Wu.
        Asu = cols(Ax, jx);
        Jsu = cols(args.Jx, jx);

        //=========================================================================================
        // Decompose the canonical form of the saddle point matrix
        //=========================================================================================

        decomposeCanonical({ dims, Hss, Hsp, Vps, Vpp, Sbsns, Sbsp });
    }

    /// Compute the right-hand side vector in the canonical saddle point problem.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto rhs(SaddlePointSolverRhs1Args args) -> void
    {
        const auto nx  = dims.nx;
        const auto ns  = dims.ns;
        const auto nu  = dims.nu;
        const auto ny  = dims.ny;
        const auto nz  = dims.nz;
        const auto nbs = dims.nbs;
        const auto nbu = dims.nbu;
        const auto nns = dims.nns;
        const auto nl  = dims.nl;

        const auto R = echelonizer.R();

        asu = args.ax(jx);
        ap = args.ap;
        az = args.az;

        aw.head(ny) = args.ay;
        aw.tail(nz) = args.az;
        aw = R * aw;

        auto awbu = aw.segment(nbs, nbu);
        auto awl  = aw.tail(nl);

        awbu.fill(0.0); // ensure residuals w.r.t. basic unstable variables are zero
        awl.fill(0.0); // ensure residuals w.r.t. linearly dependent rows in Ax are zero
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

        const auto As = W.topRows(ny).leftCols(ns);
        const auto Au = W.topRows(ny).middleCols(ns, nu);

        const auto Js = W.bottomRows(nz).leftCols(ns);
        const auto Jp = W.bottomRows(nz).rightCols(np);

        const auto R = echelonizer.R();
        const auto Rbs = R.topRows(nbs);

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        const auto xs = xsu.head(ns) = args.x(js);
        const auto xu = xsu.tail(nu) = args.x(ju);

        const auto xbs = xs.head(nbs);
        const auto xns = xs.tail(nns);

        const auto gs = g.head(ns) = args.fx(js);

        auto as   = asu.head(ns);
        auto au   = asu.tail(nu);
        auto awbs = aw.head(nbs);
        auto awbu = aw.segment(nbs, nbu);
        auto awbl = aw.tail(nl);

        as.noalias() = -(gs + tr(As) * args.y + tr(Js) * args.z);
        au.fill(0.0);

        ap = -args.v;
        az = -args.h;

        awstar.head(ny) = args.b - Au*xu;
        awstar.tail(nz) = Js*xs + Jp*args.p - args.h;

        awbs = multiplyMatrixVectorWithoutResidualRoundOffError(Rbs, awstar);
        awbs.noalias() -= xbs + Sbsns*xns + Sbsp*args.p;

        awbu.fill(0.0); // ensure residuals w.r.t. basic unstable variables are zero
        awbl.fill(0.0); // ensure residuals w.r.t. linearly dependent rows in Wx = [Ax; Jx] are zero
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

        const auto Hss = H.topLeftCorner(ns, ns);
        const auto Hsp = H.topRightCorner(ns, np);

        const auto Vps = V.leftCols(ns);
        const auto Vpp = V.rightCols(np);

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        const auto R = echelonizer.R();

        const auto as   = asu.head(ns);
        const auto au   = asu.tail(nu);
        const auto awbs = aw.head(nbs);

        auto xs  = xsu.head(ns);
        auto xu  = xsu.tail(nu);
        auto wbs = w.head(nbs);
        auto wbu = w.segment(nbs, nbu);
        auto wl  = w.tail(nl);
        auto p   = args.sp;
        auto y   = args.sy;
        auto z   = args.sz;

        wbu.fill(0.0); // zero for w values associated with unstable basic variables
        wl.fill(0.0);  // zero for w values associated with linearly dependent rows in Wx = [Ax; Jx]

        solveCanonical({ dims, Hss, Hsp, Vps, Vpp, Sbsns, Sbsp, as, ap, awbs, xs, p, wbs });

        waux.noalias() = tr(R) * w;

        args.sx(js) = xs;
        args.sx(ju) = au;
        args.sp = p;
        args.sy = waux.head(ny);
        args.sz = waux.tail(nz);
    }

    /// Multiply the saddle point matrix with a given vector.
    /// @note Ensure method @ref canonicalize has been called before this method.
    auto multiply(SaddlePointSolverMultiplyArgs args) -> void
    {
        const auto nx = dims.nx;
        const auto ns = dims.ns;
        const auto nu = dims.nu;
        const auto np = dims.np;
        const auto ny = dims.ny;
        const auto nz = dims.nz;

        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        const auto Hss = H.topLeftCorner(ns, ns);
        const auto Hsp = H.topRightCorner(ns, np);

        const auto Vpx = V.leftCols(nx);
        const auto Vpp = V.rightCols(np);
        const auto Vps = Vpx.leftCols(ns);

        const auto As = W.topRows(ny).leftCols(ns);

        const auto Js = W.bottomRows(nz).leftCols(ns);
        const auto Jp = W.bottomRows(nz).rightCols(np);

        auto as = args.ax(js);
        auto au = args.ax(ju);
        auto ap = args.ap;
        auto ay = args.ay;
        auto az = args.az;

        const auto rs = xsu.head(ns) = args.rx(js);
        const auto ru = xsu.tail(nu) = args.rx(ju);
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
        const auto ny = dims.ny;
        const auto nz = dims.nz;

        const auto js = jx.head(ns);
        const auto ju = jx.tail(nu);

        const auto Hss = H.topLeftCorner(ns, ns);
        const auto Hsp = H.topRightCorner(ns, np);

        const auto Vpx = V.leftCols(nx);
        const auto Vpp = V.rightCols(np);
        const auto Vps = Vpx.leftCols(ns);

        const auto As = W.topRows(ny).leftCols(ns);

        const auto Js = W.bottomRows(nz).leftCols(ns);
        const auto Jp = W.bottomRows(nz).rightCols(np);

        auto as = args.ax(js);
        auto au = args.ax(ju);
        auto ap = args.ap;
        auto ay = args.ay;
        auto az = args.az;

        const auto rs = xsu.head(ns) = args.rx(js);
        const auto ru = xsu.tail(nu) = args.rx(ju);
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

        const auto Hss = H.topLeftCorner(ns, ns);
        const auto Hsp = H.topRightCorner(ns, np);

        const auto As = W.topRows(ny).leftCols(ns);
        const auto Au = W.topRows(ny).middleCols(ns, nu);

        const auto Js = W.bottomRows(nz).leftCols(ns);
        const auto Jp = W.bottomRows(nz).rightCols(np);

        const auto Vps = V.leftCols(ns);
        const auto Vpp = V.rightCols(np);

        const auto Sbsns = S.topLeftCorner(nbs, nns);
        const auto Sbsp  = S.topRightCorner(nbs, np);

        const auto R = echelonizer.R();

        const auto as = asu.head(ns);
        const auto au = asu.tail(nu);

        return { dims, js, jbs, jns, ju, jbu, jnu, R, Hss, Hsp, Vps, Vpp,
            As, Au, Ap, Js, Jp, Sbsns, Sbsp, as, au, ap, ay, az, aw };
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

auto SaddlePointSolver::canonicalize(SaddlePointSolverCanonicalize2Args args) -> void
{
    pimpl->canonicalize(args);
}

auto SaddlePointSolver::decompose() -> void
{
    pimpl->decompose();
}

auto SaddlePointSolver::decompose(SaddlePointSolverDecomposeArgs args) -> void
{
    pimpl->decompose(args);
}

auto SaddlePointSolver::rhs(SaddlePointSolverRhs1Args args) -> void
{
    pimpl->rhs(args);
}

auto SaddlePointSolver::rhs(SaddlePointSolverRhs2Args args) -> void
{
    pimpl->rhs(args);
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
