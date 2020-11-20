// // Optima is a C++ library for solving linear and non-linear constrained optimization problems
// //
// // Copyright (C) 2014-2018 Allan Leal
// //
// // This program is free software: you can redistribute it and/or modify
// // it under the terms of the GNU General Public License as published by
// // the Free Software Foundation, either version 3 of the License, or
// // (at your option) any later version.
// //
// // This program is distributed in the hope that it will be useful,
// // but WITHOUT ANY WARRANTY; without even the implied warranty of
// // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// // GNU General Public License for more details.
// //
// // You should have received a copy of the GNU General Public License
// // along with this program. If not, see <http://www.gnu.org/licenses/>.

// #include "Solver.hpp"

// // Optima includes
// #include <Optima/Constants.hpp>
// #include <Optima/Exception.hpp>
// #include <Optima/IndexUtils.hpp>
// #include <Optima/MasterSolver.hpp>
// #include <Optima/Options.hpp>
// #include <Optima/Problem.hpp>
// #include <Optima/Result.hpp>
// #include <Optima/State.hpp>
// #include <Optima/Timing.hpp>
// #include <Optima/Utils.hpp>

// namespace Optima {
// namespace detail {

// auto initMasterSolver(const Problem& problem) -> MasterSolver
// {
//     const auto [nx, np, nbe, nbg, nhe, nhg] = problem.dims;

//     const auto nr   = nbg;
//     const auto ns   = nhg;
//     const auto nxrs = nx + nr + ns;
//     const auto ny   = nbe + nbg;
//     const auto nz   = nhe + nhg;

//     // Create matrix Ax = [ [Aex, 0, 0], [Agx, I, 0] ]
//     Matrix Ax = zeros(ny, nxrs);
//     Ax.leftCols(nx) << problem.Aex, problem.Agx;
//     Ax.middleCols(nx, nr).bottomRows(nr).diagonal().fill(1.0);

//     // Create matrix Ap = [ [Aep], [Agp] ]
//     Matrix Ap = zeros(ny, np);

//     if(np > 0) Ap << problem.Aep, problem.Agp;

//     return MasterSolver({ MasterDims{nxrs, np, ny, nz}, Ax, Ap });
// }

// } // namespace detail

// struct Solver::Impl
// {
//     MasterSolver mastersolver; ///< The master optimization solver.
//     Dims dims;                 ///< The dimension information of variables and constraints in the optimization problem.
//     Index nx   = 0;            ///< The number of variables x in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
//     Index nr   = 0;            ///< The number of variables r in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
//     Index ns   = 0;            ///< The number of variables s in xrs = (x, r, s) = (x, xbg, xhg) = xbar.
//     Index nxrs = 0;            ///< The number of variables in xrs = (x, xbg, xhg).
//     Index np   = 0;            ///< The number of parameter variables p.
//     Index ny   = 0;            ///< The number of Lagrange multipliers y (i.e., the dimension of vector b = (be, bg)).
//     Index nz   = 0;            ///< The number of Lagrange multipliers z (i.e., the dimension of vector h = (he, hg)).
//     Vector b;                  ///< The right-hand side vector b = (be, bg) in the master optimization problem.
//     Vector xrslower;           ///< The lower bounds of vector xrs = (x, xbg, xhg) in the master optimization problem.
//     Vector xrsupper;           ///< The upper bounds of vector xrs = (x, xbg, xhg) in the master optimization problem.
//     Indices iordering;         ///< The ordering of the variables xrs = (x, xbg, xhg) as (*stable*, *lower unstable*, *upper unstable*).

//     /// Construct a Solver instance with given optimization problem.
//     Impl(const Problem& problem)
//     : mastersolver(detail::initMasterSolver(problem)), dims(problem.dims)
//     {
//         // Initialize dimension variables
//         nx   = dims.x;
//         nr   = dims.bg;
//         ns   = dims.hg;
//         nxrs = nx + nr + ns;
//         np   = dims.p;
//         ny   = dims.be + dims.bg;
//         nz   = dims.he + dims.hg;

//         // Initialize vectors
//         xrslower.resize(nxrs);
//         xrsupper.resize(nxrs);
//         b.resize(ny);

//         // Initialize the ordering of the variables.
//         iordering = indices(nxrs);
//     }

//     /// Set the options for the optimization calculation.
//     auto setOptions(const Options& options) -> void
//     {
//         mastersolver.setOptions(options);
//     }

//     /// Solve the optimization problem.
//     auto solve(State& state, const Problem& problem) -> Result
//     {
//         // Auxiliary references
//         const auto plower = problem.plower;
//         const auto pupper = problem.pupper;

//         assert(problem.f);
//         assert(problem.dims.he == 0 || problem.he);
//         assert(problem.dims.hg == 0 || problem.hg);
//         assert(problem.dims.p == 0 || problem.v);

//         // Create the objective function for the master optimization problem
//         auto f = [&](VectorView xrs, VectorView p, ObjectiveResult res)
//         {
//             // Views to sub-vectors in xrs = (x, r, s)
//             const auto x = xrs.head(nx);
//             const auto r = xrs.segment(nx, nr);
//             const auto s = xrs.tail(ns);

//             // Views to sub-vectors in fxrs = (fx, fr, fs)
//             auto fx = res.fx.head(nx);
//             auto fr = res.fx.segment(nx, nr);
//             auto fs = res.fx.tail(ns);

//             // Views to sub-matrices in fxrsxrs = [ [fxx fxr fxs], [frx frr frs], [fsx fsr fss] ]
//             auto fxx = res.fxx.topRows(nx).leftCols(nx);
//             auto fxr = res.fxx.topRows(nx).middleCols(nx, nr);
//             auto fxs = res.fxx.topRows(nx).rightCols(ns);

//             auto frx = res.fxx.middleRows(nx, nr).leftCols(nx);
//             auto frr = res.fxx.middleRows(nx, nr).middleCols(nx, nr);
//             auto frs = res.fxx.middleRows(nx, nr).rightCols(ns);

//             auto fsx = res.fxx.bottomRows(ns).leftCols(nx);
//             auto fsr = res.fxx.bottomRows(ns).middleCols(nx, nr);
//             auto fss = res.fxx.bottomRows(ns).rightCols(ns);

//             // Views to sub-matrices in fxrsp = [ [fxp], [frp], [fsp] ]
//             auto fxp = res.fxp.topRows(nx);
//             auto frp = res.fxp.middleRows(nx, nr);
//             auto fsp = res.fxp.bottomRows(ns);

//             // Set blocks to zero, except fx, fxx, fxp (computed via the objective function next)
//             fr.fill(0.0);
//             fs.fill(0.0);
//             fxr.fill(0.0);
//             fxs.fill(0.0);
//             frx.fill(0.0);
//             frr.fill(0.0);
//             frs.fill(0.0);
//             fsx.fill(0.0);
//             fsr.fill(0.0);
//             fss.fill(0.0);
//             frp.fill(0.0);
//             fsp.fill(0.0);

//             // Use the objective function to compute f, fx, fxx, fxp
//             ObjectiveResult fres{res.f, fx, fxx, fxp, res.diagfxx};

//             return problem.f(x, p, fres);
//         };

//         // Create the non-linear equality constraint for the master optimization problem
//         auto h = [&](VectorView xrs, VectorView p, ConstraintResult res)
//         {
//             // Views to sub-vectors in xrs = (x, r, s)
//             const auto x = xrs.head(nx);
//             const auto r = xrs.segment(nx, nr);
//             const auto s = xrs.tail(ns);

//             // Views to sub-vectors in h = (he, hg)
//             auto he = res.h.topRows(dims.he);
//             auto hg = res.h.bottomRows(dims.hg);

//             // Views to sub-matrices in dh/d(xrs) = [ [he_x he_r he_s], [hg_x hg_r hg_s] ]
//             auto he_x = res.hx.topRows(dims.he).leftCols(nx);
//             auto he_r = res.hx.topRows(dims.he).middleCols(nx, nr);
//             auto he_s = res.hx.topRows(dims.he).rightCols(ns);

//             auto hg_x = res.hx.bottomRows(dims.hg).leftCols(nx);
//             auto hg_r = res.hx.bottomRows(dims.hg).middleCols(nx, nr);
//             auto hg_s = res.hx.bottomRows(dims.hg).rightCols(ns);

//             // Views to sub-matrices in dh/dp = [he_p; hg_p]
//             auto he_p = res.hp.topRows(dims.he);
//             auto hg_p = res.hp.bottomRows(dims.hg);

//             // Set all blocks to zero, except hg_s which is I and he_x and hg_x computed next via he and hg functions
//             he_r.fill(0.0);
//             he_s.fill(0.0);
//             hg_r.fill(0.0);
//             hg_s.fill(0.0);
//             hg_s.diagonal().fill(1.0);

//             ConstraintResult re{he, he_x, he_p};
//             ConstraintResult rg{hg, hg_x, hg_p};

//             if(problem.dims.he) if(problem.he(x, p, re) == FAILED) return FAILED;
//             if(problem.dims.hg) if(problem.hg(x, p, rg) == FAILED) return FAILED;

//             hg.noalias() += s;

//             return SUCCEEDED;
//         };

//         // Create the external non-linear constraint for the master optimization problem
//         auto v = [&](VectorView xrs, VectorView p, ConstraintResult res)
//         {
//             // Views to sub-vectors in xrs = (x, r, s)
//             const auto x = xrs.head(nx);
//             const auto r = xrs.segment(nx, nr);
//             const auto s = xrs.tail(ns);

//             // Views to sub-matrices in dv/d(xrs) = [ vx vr vs ]
//             auto vx = res.hx.leftCols(nx);
//             auto vr = res.hx.middleCols(nx, nr);
//             auto vs = res.hx.rightCols(ns);

//             // Auxiliary references to v and vp = dv/dp
//             auto v = res.h;
//             auto vp = res.hp;

//             // Set vr = dv/dr = 0 and vs = dv/ds = 0
//             vr.fill(0.0);
//             vs.fill(0.0);

//             // Compute v, vx, vp using the given external constraint function v(x, p)
//             ConstraintResult vres{v, vx, vp};

//             return problem.v(x, p, vres);
//         };

//         // Initialize vector with lower bounds for bar(x) = (x, xbg, xhg)
//         xrslower.head(nx) = problem.xlower;
//         xrslower.tail(nr + ns).fill(-infinity());

//         // Initialize vector with upper bounds for bar(x) = (x, xbg, xhg)
//         xrsupper.head(nx) = problem.xupper;
//         xrslower.tail(nr + ns).fill(0.0);

//         // Initialize vector b = (be, bg)
//         b << problem.be, problem.bg;

//         // Create references to state members
//         auto xbar       = state.xbar;
//         auto p          = state.p;
//         auto w          = state.w;
//         auto sbar       = state.sbar;
//         auto& stability = state.stability;

//         auto result = mastersolver.solve({ f, h, v, b, xrslower, xrsupper, {} }, { xbar, p, w });

//         return result;
//     }
// };

// Solver::Solver(const Problem& problem)
// : pimpl(new Impl(problem))
// {}

// Solver::Solver(const Solver& other)
// : pimpl(new Impl(*other.pimpl))
// {}

// Solver::~Solver()
// {}

// auto Solver::operator=(Solver other) -> Solver&
// {
//     pimpl = std::move(other.pimpl);
//     return *this;
// }

// auto Solver::setOptions(const Options& options) -> void
// {
// 	pimpl->setOptions(options);
// }

// auto Solver::solve(State& state, const Problem& problem) -> Result
// {
//     return pimpl->solve(state, problem);
// }

// } // namespace Optima
