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

#include "Solver.hpp"

// Optima includes
#include <Optima/BasicSolver.hpp>
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/Options.hpp>
#include <Optima/Problem.hpp>
#include <Optima/Result.hpp>
#include <Optima/State.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {
namespace detail {

auto initBasicSolver(const Problem& problem) -> BasicSolver
{
    const auto [nx, np, mbe, mbg, mhe, mhg] = problem.dims;

    const auto nr   = mbg;
    const auto ns   = mhg;
    const auto nxrs = nx + nr + ns;
    const auto mb   = mbe + mbg;
    const auto mh   = mhe + mhg;
    const auto m    = mb + mh;

    // Create matrix Ax = [ [Aex, 0, 0], [Agx, I, 0] ]
    Matrix Ax = zeros(mb, nxrs);
    Ax.leftCols(nx) << problem.Aex, problem.Agx;
    Ax.middleCols(nx, nr).bottomRows(nr).diagonal().fill(1.0);

    // Create matrix Ap = [ [Aep], [Agp] ]
    Matrix Ap = zeros(mb, np);

    if(np > 0) Ap << problem.Aep, problem.Agp;

    return BasicSolver({ nxrs, np, m, Ax, Ap });
}

} // namespace detail

struct Solver::Impl
{
    BasicSolver basicsolver; ///< The basic optimization solver.
    Dims dims;               ///< The dimension information of variables and constraints in the optimization problem.
    Index nx   = 0;          ///< The number of variables x in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
    Index nr   = 0;          ///< The number of variables r in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
    Index ns   = 0;          ///< The number of variables s in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
    Index nxrs = 0;          ///< The number of variables in xrs = (x, xbg, xhg).
    Index np   = 0;          ///< The number of parameter variables p.
    Index mb   = 0;          ///< The dimension of vector b = (be, bg).
    Index mh   = 0;          ///< The dimension of vector h = (he, hg).
    Index m    = 0;          ///< The dimension of number m = mb + mh.
    Vector b;                ///< The right-hand side vector b = (be, bg) in the basic optimization problem.
    Vector xrslower;         ///< The lower bounds of vector xrs = (x, xbg, xhg) in the basic optimization problem.
    Vector xrsupper;         ///< The upper bounds of vector xrs = (x, xbg, xhg) in the basic optimization problem.
    Indices iordering;       ///< The ordering of the variables xrs = (x, xbg, xhg) as (*stable*, *lower unstable*, *upper unstable*).
    IndexNumber nlu;         ///< The number of lower unstable variables in xrs = (x, xbg, xhg).
    IndexNumber nuu;         ///< The number of upper unstable variables in xrs = (x, xbg, xhg).

    /// Construct a Solver instance with given optimization problem.
    Impl(const Problem& problem)
    : basicsolver(detail::initBasicSolver(problem)), dims(problem.dims)
    {
        // Initialize dimension variables
        nx   = dims.x;
        nr   = dims.bg;
        ns   = dims.hg;
        nxrs = nx + nr + ns;
        np   = dims.p;
        mb   = dims.be + dims.bg;
        mh   = dims.he + dims.hg;
        m    = mb + mh;

        // Initialize vectors
        xrslower.resize(nxrs);
        xrsupper.resize(nxrs);
        b.resize(m);

        // Initialize the ordering of the variables.
        iordering = indices(nxrs);
    }

    /// Set the options for the optimization calculation.
    auto setOptions(const Options& options) -> void
    {
        basicsolver.setOptions(options);
    }

    /// Solve the optimization problem.
    auto solve(State& state, const Problem& problem) -> Result
    {
        // Auxiliary references
        const auto plower = problem.plower;
        const auto pupper = problem.pupper;

        assert(problem.f);
        assert(problem.dims.he == 0 || problem.he);
        assert(problem.dims.hg == 0 || problem.hg);

        // Create the objective function for the basic problem
        auto f = [&](VectorConstRef xrs, VectorConstRef p, ObjectiveResult& res)
        {
            // Views to sub-vectors in xrs = (x, r, s)
            const auto x = xrs.head(nx);
            const auto r = xrs.segment(nx, nr);
            const auto s = xrs.tail(ns);

            // Views to sub-vectors in fxrs = (fx, fr, fs)
            auto fx = res.fx.head(nx);
            auto fr = res.fx.segment(nx, nr);
            auto fs = res.fx.tail(ns);

            // Views to sub-matrices in fxrsxrs = [ [fxx fxr fxs], [frx frr frs], [fsx fsr fss] ]
            auto fxx = res.fxx.topRows(nx).leftCols(nx);
            auto fxr = res.fxx.topRows(nx).middleCols(nx, nr);
            auto fxs = res.fxx.topRows(nx).rightCols(ns);

            auto frx = res.fxx.middleRows(nx, nr).leftCols(nx);
            auto frr = res.fxx.middleRows(nx, nr).middleCols(nx, nr);
            auto frs = res.fxx.middleRows(nx, nr).rightCols(ns);

            auto fsx = res.fxx.bottomRows(ns).leftCols(nx);
            auto fsr = res.fxx.bottomRows(ns).middleCols(nx, nr);
            auto fss = res.fxx.bottomRows(ns).rightCols(ns);

            // Views to sub-matrices in fxrsp = [ [fxp], [frp], [fsp] ]
            auto fxp = res.fxp.topRows(nx);
            auto frp = res.fxp.middleRows(nx, nr);
            auto fsp = res.fxp.bottomRows(ns);

            // Set blocks to zero, except fx, fxx, fxp (computed via the objective function next)
            fr.fill(0.0);
            fs.fill(0.0);
            fxr.fill(0.0);
            fxs.fill(0.0);
            frx.fill(0.0);
            frr.fill(0.0);
            frs.fill(0.0);
            fsx.fill(0.0);
            fsr.fill(0.0);
            fss.fill(0.0);
            frp.fill(0.0);
            fsp.fill(0.0);

            // Use the objective function to compute f, fx, fxx, fxp
            ObjectiveResult fres{res.f, fx, fxx, fxp};
            fres.requires = res.requires;

            problem.f(x, p, fres);

            res.failed = fres.failed;
        };

        // Create the non-linear equality constraint for the basic problem
        auto h = [&](VectorConstRef xrs, VectorConstRef p, ConstraintResult& res)
        {
            // Views to sub-vectors in xrs = (x, r, s)
            const auto x = xrs.head(nx);
            const auto r = xrs.segment(nx, nr);
            const auto s = xrs.tail(ns);

            // Views to sub-vectors in h = (he, hg)
            auto he = res.h.topRows(dims.he);
            auto hg = res.h.bottomRows(dims.hg);

            // Views to sub-matrices in dh/d(xrs) = [ [he_x he_r he_s], [hg_x hg_r hg_s] ]
            auto he_x = res.hx.topRows(dims.he).leftCols(nx);
            auto he_r = res.hx.topRows(dims.he).middleCols(nx, nr);
            auto he_s = res.hx.topRows(dims.he).rightCols(ns);

            auto hg_x = res.hx.bottomRows(dims.hg).leftCols(nx);
            auto hg_r = res.hx.bottomRows(dims.hg).middleCols(nx, nr);
            auto hg_s = res.hx.bottomRows(dims.hg).rightCols(ns);

            // Views to sub-matrices in dh/dp = [he_p; hg_p]
            auto he_p = res.hp.topRows(dims.he);
            auto hg_p = res.hp.bottomRows(dims.hg);

            // Set all blocks to zero, except hg_s which is I and he_x and hg_x computed next via he and hg functions
            he_r.fill(0.0);
            he_s.fill(0.0);
            hg_r.fill(0.0);
            hg_s.fill(0.0);
            hg_s.diagonal().fill(1.0);

            ConstraintResult re{he, he_x, he_p};
            ConstraintResult rg{hg, hg_x, hg_p};

            if(problem.dims.he) problem.he(x, p, re);
            if(problem.dims.hg) problem.hg(x, p, rg);

            hg.noalias() += s;

            res.failed = re.failed || rg.failed;
        };

        // Create the external non-linear constraint for the basic problem
        auto v = [&](VectorConstRef xrs, VectorConstRef p, ConstraintResult& res)
        {
            // Views to sub-vectors in xrs = (x, r, s)
            const auto x = xrs.head(nx);
            const auto r = xrs.segment(nx, nr);
            const auto s = xrs.tail(ns);

            // Views to sub-matrices in dv/d(xrs) = [ vx vr vs ]
            auto vx = res.hx.leftCols(nx);
            auto vr = res.hx.middleCols(nx, nr);
            auto vs = res.hx.rightCols(ns);

            // Auxiliary references to v and vp = dv/dp
            auto v = res.h;
            auto vp = res.hp;

            // Set vr = dv/dr = 0 and vs = dv/ds = 0
            vr.fill(0.0);
            vs.fill(0.0);

            // Compute v, vx, vp using the given external constraint function v(x, p)
            ConstraintResult vres{v, vx, vp};

            problem.v(x, p, vres);

            res.failed = vres.failed;
        };

        // Initialize vector with lower bounds for bar(x) = (x, xbg, xhg)
        xrslower.head(nx) = problem.xlower;
        xrslower.tail(nr + ns).fill(-infinity());

        // Initialize vector with upper bounds for bar(x) = (x, xbg, xhg)
        xrsupper.head(nx) = problem.xupper;
        xrslower.tail(nr + ns).fill(0.0);

        // Initialize vector b = (be, bg)
        b << problem.be, problem.bg;

        // Create references to state members
        auto xbar       = state.xbar;
        auto y          = state.y;
        auto p          = state.p;
        auto zbar       = state.zbar;
        auto& stability = state.stability;

        // Solve the constructed basic optimization problem
        auto result = basicsolver.solve({ f, h, v, b, xrslower, xrsupper, plower, pupper, xbar, p, y, zbar, stability });

        return result;
    }
};

Solver::Solver(const Problem& problem)
: pimpl(new Impl(problem))
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

auto Solver::solve(State& state, const Problem& problem) -> Result
{
    return pimpl->solve(state, problem);
}

} // namespace Optima
