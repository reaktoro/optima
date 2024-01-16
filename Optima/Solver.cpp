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

#include "Solver.hpp"

// Optima includes
#include <Optima/Constants.hpp>
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/MasterSolver.hpp>
#include <Optima/Options.hpp>
#include <Optima/Problem.hpp>
#include <Optima/Result.hpp>
#include <Optima/Sensitivity.hpp>
#include <Optima/State.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct Solver::Impl
{
    Dims dims;                      ///< The dimensions of the variables and constraints in the optimization problem.
    Options options;                ///< The options for the optimization problem.
    MasterSolver msolver;           ///< The master optimization solver.
    MasterProblem mproblem;         ///< The master optimization problem.
    MasterState mstate;             ///< The master optimization state.
    MasterSensitivity msensitivity; ///< The sensitivity derivatives of the master optimization state.
    Index nx    = 0;                ///< The number of variables x in xbar = (x, xbg, xhg).
    Index nxbg  = 0;                ///< The number of variables xbg in xbar = (x, xbg, xhg).
    Index nxhg  = 0;                ///< The number of variables xhg in xbar = (x, xbg, xhg).
    Index nxbar = 0;                ///< The number of variables in xbar = (x, xbg, xhg).
    Index np    = 0;                ///< The number of parameter variables p.
    Index ny    = 0;                ///< The number of Lagrange multipliers y (i.e., the dimension of vector b = (be, bg)).
    Index nz    = 0;                ///< The number of Lagrange multipliers z (i.e., the dimension of vector h = (he, hg)).
    Index nwbar = 0;                ///< The number of Lagrange multipliers in wbar = (ye, yg, ze, zg).
    Vector xbarlower;               ///< The lower bounds of vector xbar = (x, xbg, xhg) in the master optimization problem.
    Vector xbarupper;               ///< The upper bounds of vector xbar = (x, xbg, xhg) in the master optimization problem.

    /// Construct a Solver default instance.
    Impl()
    {
    }

    /// Set the options for the optimization calculation.
    auto setOptions(const Options& opts) -> void
    {
        options = opts;
    }

    /// Update the options for the master optimization problem.
    auto updateMasterOptions() -> void
    {
        if(options.output.active)
        {
            // Check size conformance for given variable names
            errorif(!options.output.xnames.empty() && options.output.xnames.size() != nx, "Expecting ", nx, " primal variable names in options.output.xnames, but got ", options.output.xnames.size(), " instead.");
            errorif(!options.output.xbgnames.empty() && options.output.xbgnames.size() != nxbg, "Expecting ", nxbg, " primal slack variable names in options.output.xbgnames, but got ", options.output.xbgnames.size(), " instead.");
            errorif(!options.output.xhgnames.empty() && options.output.xhgnames.size() != nxhg, "Expecting ", nxhg, " primal slack variable names in options.output.xhgnames, but got ", options.output.xhgnames.size(), " instead.");

            Options opts(options);

            // Clear incoming variable names because they will be set below for the MasterSolver.
            opts.output.xnames.clear();
            opts.output.xbgnames.clear();
            opts.output.xhgnames.clear();

            // Add x variable names into options.output.xnames
            if(options.output.xnames.empty())
                for(auto i = 0; i < nx; ++i)
                    opts.output.xnames.push_back(std::to_string(i));
            else opts.output.xnames = options.output.xnames;

            // Add xbg variable names into options.output.xnames
            if(options.output.xbgnames.empty())
                for(auto i = 0; i < nxbg; ++i)
                    opts.output.xnames.push_back("bg:" + std::to_string(i)); // x[bg:0], x[bg:1] for slack variables associated to linear inequality constraints
            else opts.output.xnames.insert(opts.output.xnames.end(), options.output.xbgnames.begin(), options.output.xbgnames.end());

            // Add xhg variable names into options.output.xnames
            if(options.output.xhgnames.empty())
                for(auto i = 0; i < nxhg; ++i)
                    opts.output.xnames.push_back("hg:" + std::to_string(i)); // x[hg:0], x[hg:1] for slack variables associated to non-linear inequality constraints
            else opts.output.xnames.insert(opts.output.xnames.end(), options.output.xhgnames.begin(), options.output.xhgnames.end());

            msolver.setOptions(opts);
        }
        else msolver.setOptions(options);
    }

    /// Update the master problem object `mproblem` with given Problem object.
    auto updateMasterProblem(const Problem& problem) -> void
    {
        // Initialize dimension variables
        dims  = problem.dims;
        nx    = dims.x;
        nxbg  = dims.bg;
        nxhg  = dims.hg;
        nxbar = nx + nxbg + nxhg;
        np    = dims.p;
        ny    = dims.be + dims.bg;
        nz    = dims.he + dims.hg;
        nwbar = ny + nz;

        errorif(!problem.f.initialized(),
            "Cannot solve the optimization problem. "
            "You have not initialized the objective function. "
            "Ensure Problem::f is properly initialized.");

        errorif(dims.he > 0 && !problem.he.initialized(),
            "Cannot solve the optimization problem. "
            "You have not initialized the constraint function he(x, p). "
            "Ensure Problem::he is properly initialized.");

        errorif(dims.hg > 0 && !problem.hg.initialized(),
            "Cannot solve the optimization problem. "
            "You have not initialized the constraint function hg(x, p). "
            "Ensure Problem::hg is properly initialized.");

        errorif(dims.p > 0 && !problem.v.initialized(),
            "Cannot solve the optimization problem. "
            "You have not initialized the complementary constraint function v(x, p). "
            "Ensure Problem::v is properly initialized.");

        // Initialize the dimensions of the master optimization problem
        mproblem.dims = MasterDims(nxbar, np, ny, nz);

        // Initialize the resources function in the master optimization problem
        mproblem.r = problem.r;

        // Create the objective function for the master optimization problem
        mproblem.f = [&](ObjectiveResultRef resbar, VectorView xbar, VectorView p, VectorView c, ObjectiveOptions opts)
        {
            resbar.fx.fill(0.0);
            resbar.fxx.fill(0.0);
            resbar.fxp.fill(0.0);
            resbar.fxc.fill(0.0);

            auto x   = xbar.head(nx);
            auto fx  = resbar.fx.head(nx);
            auto fxx = resbar.fxx.topLeftCorner(nx, nx);
            auto fxp = resbar.fxp.topRows(nx);
            auto fxc = resbar.fxc.topRows(nx);

            ObjectiveResultRef fres(resbar.f, fx, fxx, fxp, fxc, resbar.diagfxx, resbar.fxx4basicvars, resbar.succeeded);

            problem.f(fres, x, p, c, opts);
        };

        // Create the non-linear equality constraint for the master optimization problem
        mproblem.h = [&](ConstraintResultRef resbar, VectorView xbar, VectorView p, VectorView c, ConstraintOptions opts)
        {
            // Views to sub-vectors in xbar = (x, xbg, xhg)
            const auto x   = xbar.head(nx);
            const auto xbg = xbar.segment(nx, nxbg);
            const auto xhg = xbar.tail(nxhg);

            // Views to sub-vectors in h = (he, hg)
            auto he = resbar.val.head(dims.he);
            auto hg = resbar.val.tail(dims.hg);

            // Views to sub-matrices in dh/d(xbar) = [ [dhe/dx dhe/dxbg dhe/dxhg], [dhg/dx dhg/dxbg dhg/dxhg] ]
            auto he_x   = resbar.ddx.topRows(dims.he).leftCols(nx);
            auto he_xbg = resbar.ddx.topRows(dims.he).middleCols(nx, nxbg);
            auto he_xhg = resbar.ddx.topRows(dims.he).rightCols(nxhg);

            auto hg_x   = resbar.ddx.bottomRows(dims.hg).leftCols(nx);
            auto hg_xbg = resbar.ddx.bottomRows(dims.hg).middleCols(nx, nxbg);
            auto hg_xhg = resbar.ddx.bottomRows(dims.hg).rightCols(nxhg);

            // Views to sub-matrices in dh/dp = [he_p; hg_p]
            auto he_p = resbar.ddp.topRows(dims.he);
            auto hg_p = resbar.ddp.bottomRows(dims.hg);

            // Views to sub-matrices in dh/dc = [he_c; hg_c]
            auto he_c = resbar.ddc.topRows(dims.he);
            auto hg_c = resbar.ddc.bottomRows(dims.hg);

            // Set all blocks related to xbg and xhg to zero, except hg_xhg which is identity
            he_xbg.fill(0.0);
            he_xhg.fill(0.0);
            hg_xbg.fill(0.0);
            hg_xhg.fill(0.0);
            hg_xhg.diagonal().fill(1.0);

            ConstraintResultRef heres(he, he_x, he_p, he_c, resbar.ddx4basicvars, resbar.succeeded);

            problem.he(heres, x, p, c, opts);

            ConstraintResultRef hgres(hg, hg_x, hg_p, hg_c, resbar.ddx4basicvars, resbar.succeeded);

            problem.hg(hgres, x, p, c, opts);

            hg.noalias() += xhg;
        };

        // Create the external non-linear constraint for the master optimization problem
        mproblem.v = [&](ConstraintResultRef res, VectorView xbar, VectorView p, VectorView c, ConstraintOptions opts)
        {
            // Views to sub-vectors in xbar = (x, xbg, xhg)
            const auto x   = xbar.head(nx);
            const auto xbg = xbar.segment(nx, nxbg);
            const auto xhg = xbar.tail(nxhg);

            // Views to sub-matrices in dv/d(xbar) = [ dv/dx dv/dxbg dv/dxhg ]
            auto v_x   = res.ddx.leftCols(nx);
            auto v_xbg = res.ddx.middleCols(nx, nxbg);
            auto v_xhg = res.ddx.rightCols(nxhg);

            // Auxiliary references to v, vp = dv/dp, vc = dv/dc
            auto v   = res.val;
            auto v_p = res.ddp;
            auto v_c = res.ddc;

            // Set dv/dxbg = 0 and dv/dxhg = 0
            v_xbg.fill(0.0);
            v_xhg.fill(0.0);

            ConstraintResultRef vres(v, v_x, v_p, v_c, res.ddx4basicvars, res.succeeded);

            problem.v(vres, x, p, c, opts);
        };

        // Initialize the lower bounds of xbar = (x, xbg, xhg)
        xbarlower.resize(nxbar);
        xbarlower.head(nx) = problem.xlower;
        xbarlower.tail(nxbg + nxhg).fill(-infinity());

        // Initialize the upper bounds of xbar = (x, xbg, xhg)
        xbarupper.resize(nxbar);
        xbarupper.head(nx) = problem.xupper;
        xbarupper.tail(nxbg + nxhg).fill(0.0);

        // Initialize vector b = (be, bg)
        mproblem.b.resize(ny);
        mproblem.b << problem.be, problem.bg;

        // Initialize lower and upper bounds for x in the master problem
        mproblem.xlower = xbarlower;
        mproblem.xupper = xbarupper;

        // Initialize lower and upper bounds for p in the master problem
        mproblem.plower = problem.plower;
        mproblem.pupper = problem.pupper;

        // Initialize matrix Ax = [ [Aex, 0, 0], [Agx, I, 0] ] in the master problem
        mproblem.Ax.resize(ny, nxbar);
        if(mproblem.Ax.size()) {
            mproblem.Ax.leftCols(nx) << problem.Aex, problem.Agx;
            mproblem.Ax.middleCols(nx, nxbg).bottomRows(nxbg) = identity(nxbg, nxbg);
            mproblem.Ax.rightCols(nxhg).fill(0.0);
        }

        // Initialize matrix Ap = [ [Aep], [Agp] ] in the master problem
        mproblem.Ap.resize(ny, np);
        if(mproblem.Ap.size())
            mproblem.Ap << problem.Aep, problem.Agp;

        // Initialize the sensitivity parameters *c* in the master problem
        mproblem.c = problem.c;

        // Initialize the Jacobian matrix of *b* with respect to the sensitivity parameters *c*.
        mproblem.bc.resize(ny, dims.c);
        mproblem.bc.topRows(dims.be) = problem.bec;
        mproblem.bc.bottomRows(dims.bg) = problem.bgc;
    }

    /// Update the master state object `mstate` with given State object.
    auto updateMasterState(const State& state) -> void
    {
        // Initialize xbar = (x, xbg, xhg)
        mstate.u.x.resize(nxbar);
        mstate.u.x << state.x, state.xbg, state.xhg;

        // Initialize wbar = (ye, yg, ze, zg)
        mstate.u.w.resize(nwbar);
        mstate.u.w << state.ye, state.yg, state.ze, state.zg;

        // Initialize pbar = p
        mstate.u.p = state.p;
    }

    /// Update the given State object with computed MasterState object `mstate`.
    auto updateState(State& state) -> void
    {
        state.x   = mstate.u.x.head(nx);
        state.xbg = mstate.u.x.segment(nx, nxbg);
        state.xhg = mstate.u.x.tail(nxhg);
        state.ye  = mstate.u.w.head(ny).head(dims.be);
        state.yg  = mstate.u.w.head(ny).tail(dims.bg);
        state.ze  = mstate.u.w.tail(nz).head(dims.he);
        state.zg  = mstate.u.w.tail(nz).tail(dims.hg);
        state.p   = mstate.u.p;
        state.s   = mstate.s.head(nx);

        auto const is_xbg_or_xhg = [=](Index i) { return i >= nx; };
        std::function<Index(IndicesRef)> move_right_xbg_xhg_1 = [=](IndicesRef indices) -> Index { return moveRightIf(indices, is_xbg_or_xhg); };
        std::function<Index(IndicesRef)> move_right_xbg_xhg_2 = [=](IndicesRef indices) -> Index { return indices.size(); };
        auto move_right_xbg_xhg = nxbg + nxhg > 0 ? move_right_xbg_xhg_1 : move_right_xbg_xhg_2;

        Index const ks  = move_right_xbg_xhg(mstate.js);  // Move indices corresponding to variables xbg and xhg to the end of js.
        Index const ku  = move_right_xbg_xhg(mstate.ju);  // Move indices corresponding to variables xbg and xhg to the end of ju.
        Index const klu = move_right_xbg_xhg(mstate.jlu); // Move indices corresponding to variables xbg and xhg to the end of jlu.
        Index const kuu = move_right_xbg_xhg(mstate.juu); // Move indices corresponding to variables xbg and xhg to the end of juu.
        Index const kb  = move_right_xbg_xhg(mstate.jb);  // Move indices corresponding to variables xbg and xhg to the end of jb.
        Index const kn  = move_right_xbg_xhg(mstate.jn);  // Move indices corresponding to variables xbg and xhg to the end of jn.

        state.js  = mstate.js.head(ks);
        state.ju  = mstate.ju.head(ku);
        state.jlu = mstate.jlu.head(klu);
        state.juu = mstate.juu.head(kuu);
        state.jb  = mstate.jb.head(kb);
        state.jn  = mstate.jn.head(kn);
    }

    /// Update the given Sensitivity object with computed MasterSensitivity object `msensitivity`.
    auto updateSensitivity(Sensitivity& sensitivity) -> void
    {
        sensitivity.resize(dims);
        sensitivity.xc   = msensitivity.xc.topRows(nx);
        sensitivity.pc   = msensitivity.pc;
        sensitivity.xbgc = msensitivity.xc.middleRows(nx, nxbg);
        sensitivity.xhgc = msensitivity.xc.bottomRows(nxhg);
        sensitivity.yec  = msensitivity.wc.topRows(ny).topRows(dims.be);
        sensitivity.ygc  = msensitivity.wc.topRows(ny).bottomRows(dims.bg);
        sensitivity.zec  = msensitivity.wc.bottomRows(nz).topRows(dims.he);
        sensitivity.zgc  = msensitivity.wc.bottomRows(nz).bottomRows(dims.hg);
        sensitivity.sc   = msensitivity.sc.topRows(nx);
    }

    /// Solve the optimization problem.
    auto solve(const Problem& problem, State& state) -> Result
    {
        updateMasterProblem(problem);
        updateMasterOptions();
        updateMasterState(state);
        const auto result = msolver.solve(mproblem, mstate);
        updateState(state);
        return result;
    }

    /// Solve the optimization problem and compute the sensitivity derivatives at the end.
    auto solve(const Problem& problem, State& state, Sensitivity& sensitivity) -> Result
    {
        updateMasterProblem(problem);
        updateMasterOptions();
        updateMasterState(state);
        const auto result = msolver.solve(mproblem, mstate, msensitivity);
        updateState(state);
        updateSensitivity(sensitivity);
        return result;
    }
};

Solver::Solver()
: pimpl(new Impl())
{}

Solver::Solver(const Solver& other)
: pimpl(new Impl(*other.pimpl))
{}

Solver::~Solver()
{}

auto Solver::operator=(Solver other) -> Solver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Solver::setOptions(const Options& options) -> void
{
	pimpl->setOptions(options);
}

auto Solver::solve(const Problem& problem, State& state) -> Result
{
    return pimpl->solve(problem, state);
}

auto Solver::solve(const Problem& problem, State& state, Sensitivity& sensitivity) -> Result
{
    return pimpl->solve(problem, state, sensitivity);
}

} // namespace Optima
