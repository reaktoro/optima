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

#include "Stepper.hpp"

// C++ includes
#include <cassert>

// Optima includes
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/SaddlePointSolver.hpp>
#include <Optima/Options.hpp>
#include <Optima/StabilityChecker.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

using std::abs;

struct Stepper::Impl
{
    Options options;                   ///< The options for the optimization calculation
    Matrix Ax;                         ///< The coefficient matrix Ax in the linear equality constraints.
    Matrix Ap;                         ///< The coefficient matrix Ap in the linear equality constraints.
    Matrix hx;                         ///< The evaluated Jacobian of the equality constraint function *h(x, p)* with respect to *x*.
    Vector s;                          ///< The stability measures of the variables defined as s = g + tr(Ax)*y + tr(hx)*z.
    Vector w;                          ///< The weights of the variables used to determine which ones are basic/nonbasic.
    Vector xbar;                       ///< The solution vector x in the saddle point problem.
    Vector pbar;                       ///< The solution vector p in the saddle point problem.
    Vector ybar;                       ///< The solution vector y in the saddle point problem.
    Vector zbar;                       ///< The solution vector z in the saddle point problem.
    Vector bprime;                     ///< The auxiliary vector b' = R*[b, hx*x + h] used to compute feasibility residuals.
    Index nx = 0;                      ///< The number of primal variables in x.
    Index np = 0;                      ///< The number of parameter variables in p.
    Index ny = 0;                      ///< The number of Lagrange multipliers in y.
    Index nz = 0;                      ///< The number of Lagrange multipliers in z.
    StabilityChecker stabilitychecker; ///< The stability checker of the primal variables
    SaddlePointSolver spsolver;        ///< The saddle point solver.

    /// Construct a Stepper::Impl instance.
    Impl(StepperInitArgs args)
    : nx(args.nx), np(args.np), ny(args.ny), nz(args.nz), Ax(args.Ax), Ap(args.Ap),
      stabilitychecker({args.nx, args.np, args.ny, args.nz, args.Ax, args.Ap}),
      spsolver({args.nx, args.np, args.ny, args.nz, args.Ax, args.Ap})
    {
        // Ensure the step calculator is initialized with a positive number of primal variables.
        assert(nx > 0 && "Could not proceed with Stepper initialization. "
            "The number of primal variables x is zero.");

        // Initialize auxiliary vectors
        s = zeros(nx);
        xbar = zeros(nx);
        pbar = zeros(np);
        ybar = zeros(ny);
        zbar = zeros(nz);
        bprime = zeros(ny);
    }

    /// Initialize the step calculator before calling decompose multiple times.
    auto initialize(StepperInitializeArgs args) -> void
    {
        // Ensure consistent dimensions of vectors/matrices.
        assert(args.b.rows() == ny);
        assert(args.xlower.rows() == nx);
        assert(args.xupper.rows() == nx);
        assert(args.plower.rows() == np);
        assert(args.pupper.rows() == np);
        assert(args.x.rows() == nx);

        // Auxiliary const references
        const auto b      = args.b;
        const auto xlower = args.xlower;
        const auto xupper = args.xupper;
        const auto plower = args.plower;
        const auto pupper = args.pupper;

        // Initialize the stability checker.
        // Identify the strictly lower/upper unstable variables.
        stabilitychecker.initialize({ b, xlower, xupper, plower, pupper });

        // Set the output Stability object
        args.stability = stabilitychecker.stability();

        // Get the indices of the strictly lower and upper unstable variables
        const auto jslu = args.stability.indicesStrictlyLowerUnstableVariables();
        const auto jsuu = args.stability.indicesStrictlyUpperUnstableVariables();

        // Attach the strictly unstable variables to either their upper or lower bounds
        args.x(jsuu) = xupper(jsuu);
        args.x(jslu) = xlower(jslu);
    }

    /// Canonicalize the matrix *A = [Ax Ap]* in the saddle point matrix.
    auto canonicalize(StepperCanonicalizeArgs args) -> void
    {
        // Ensure the step calculator has been initialized.
        assert(nx != 0 && "Could not proceed with Stepper::canonicalize. "
            "Stepper object has not been initialized.");

        // Auxiliary references
        const auto x      = args.x;
        const auto p      = args.p;
        const auto y      = args.y;
        const auto z      = args.z;
        const auto fx     = args.fx;
        const auto fxx    = args.fxx;
        const auto fxp    = args.fxp;
        const auto vx     = args.vx;
        const auto vp     = args.vp;
        const auto hp     = args.hp;
        const auto xlower = args.xlower;
        const auto xupper = args.xupper;
        const auto plower = args.plower;
        const auto pupper = args.pupper;
        auto& stability   = args.stability;

        // Update the Jacobian of the equality constraint function h(x, p) with respect to x
        hx = args.hx;

        // Ensure consistent dimensions of vectors/matrices.
        assert(  x.rows()    == nx);
        assert(  p.rows()    == np);
        assert(  y.rows()    == ny);
        assert(  z.rows()    == nz);
        assert( fx.rows()    == nx);
        assert(fxx.rows()    == nx);
        assert(fxx.cols()    == nx);
        assert(fxp.rows()    == nx);
        assert(fxp.cols()    == np);
        assert( vx.rows()    == np);
        assert( vx.cols()    == nx);
        assert( vp.rows()    == np);
        assert( vp.cols()    == np);
        assert( hx.rows()    == nz);
        assert( hx.cols()    == nx);
        assert( hp.rows()    == nz);
        assert( hp.cols()    == np);
        assert(xlower.rows() == nx);
        assert(xupper.rows() == nx);
        assert(plower.rows() == np);
        assert(pupper.rows() == np);

        // Update the stability state of the primal variables
        stabilitychecker.update({ x, y, z, fx, hx, xlower, xupper });

        // Set the output Stability object
        stability = stabilitychecker.stability();

        // The indices of all unstable variables. These will be classified as
        // fixed variables when solving the saddle point problem to compute the
        // Newton step. This effectively reduces the dimension of the linear
        // algebra problem solved (i.e. the unstable variables do not make up
        // to the final dimension of the matrix equations solved).
        const auto ju = stability.indicesUnstableVariables();

        // Compute the weights used to determine basic/nonbasic variables
        const auto fakezero = 1.491667e-154; // === sqrt(double-min) where double-min = 2.22507e-308
        w = fxx.diagonal();
        w.array() += fakezero; // replaces zeros by fakezero to avoid division by zero next
        w.noalias() = abs(x - fx.cwiseQuotient(w)); /// w = |x - inv(diag(H')) * g|

        // Decompose the saddle point matrix.
        // This decomposition is later used in method solve, possibly many
        // times! Consider lower/upper unstable variables as "fixed" variables
        // in the saddle point problem. Reason: we do not need to compute
        // Newton steps for the currently unstable variables!
        spsolver.canonicalize({ fxx, fxp, vx, vp, hx, hp, ju, w });
    }

    /// Calculate the current optimality and feasibility residuals.
    /// @note Ensure method @ref canonicalize is called first.
    auto residuals(StepperResidualsArgs args) -> void
    {
        // Auxiliary references
        auto [x, p, y, z, b, h, v, fx, hx, rx, rp, ry, rz, ex, ep, ey, ez, s] = args;

        // Get a reference to the stability state of the variables
        const auto& stability = stabilitychecker.stability();

        // The indices of stable and unstable variables
        auto js = stability.indicesStableVariables();
        auto ju = stability.indicesUnstableVariables();

        // The indices of all strictly unstable variables
        auto jsu = stability.indicesStrictlyUnstableVariables();

        // Ensure consistent dimensions of vectors/matrices.
        assert(  x.rows() == nx );
        assert(  p.rows() == np );
        assert(  y.rows() == ny );
        assert(  z.rows() == nz );
        assert(  b.rows() == ny );
        assert(  h.rows() == nz );
        assert( fx.rows() == nx );
        assert( rx.rows() == nx );
        assert( rp.rows() == np );
        assert( ry.rows() == ny );
        assert( rz.rows() == nz );
        assert( ex.rows() == nx );
        assert( ep.rows() == np );
        assert( ey.rows() == ny );
        assert( ez.rows() == nz );
        assert(  s.rows() == nx );

        //======================================================================
        // Compute the canonical feasibility residuals using xb + S*xn - b' = 0
        //======================================================================
        // The computation logic below is aimed at producing feasibility
        // residuals less affected by round-off errors. For certain
        // applications (e.g. chemical equilibrium), in which some variables
        // attain very small values (e.g. H2, O2), and there might be an
        // algebraic relation between them (e.g. x[H2] = 2*x[O2] when only H2O,
        // H+, OH-, H2, O2 are considered) that strongly affects the
        // first-order optimality equations, this strategy is important.
        //======================================================================

        // Ensure the strictly unstable variables are ignored for feasibility
        // residuals. It is like if they were excluded from the computation,
        // but their final values forced to their bounds.

        // Use rx as a workspace for x' where x'[i] = 0 if i in jsu else x[i]
        auto xprime = rx;

        xprime = x;
        xprime(jsu).fill(0.0);

        spsolver.rhs({ fx, xprime, p, y, z, v, h, b });

        const auto spstate = spsolver.state();

        rx(js) = abs(spstate.as);
        rx(ju) = abs(spstate.au);
        rp = abs(spstate.ap);
        ry = abs(spstate.ay);
        rz = abs(spstate.az);

        ex = rx.array() / (1 + fx.array().abs());
        ep = rp;
        ey = ry;
        ez = rz;

        const auto dims = spstate.dims;
        const auto nbs  = dims.nbs;
        const auto nbu  = dims.nbu;
        const auto nl   = dims.nl;
        const auto jbs  = spstate.jbs;

        const auto rybs = ry.head(nbs);
        const auto rybu = ry.segment(nbs, nbu);
        const auto ryl  = ry.tail(nl);

        auto eybs = ey.head(nbs);
        auto eybu = ey.segment(nbs, nbu);
        auto eyl  = ey.tail(nl);

        const auto xbs = x(jbs);

        eybs.noalias() = rybs.cwiseQuotient((xbs.array() != 0.0).select(xbs, 1.0));

        // spsolver.residuals({ xprime, p, b, h, ry, ey });

        // Compute the relative errors of the linear/nonlinear feasibility conditions.

        //======================================================================
        // Compute the optimality residuals using g + tr(Ax)*y + tr(hx)*z = 0
        //======================================================================

        // Calculate the stability measure of the variables.
        s.noalias() = fx + tr(Ax)*y + tr(hx)*z;

        // // Calculate the residuals of the first-order optimality conditions
        // rx = s.array().abs();

        // // Set residuals with respect to unstable variables to zero. This
        // // ensures that they are not taken into account when checking for
        // // convergence.
        // rx(ju).fill(0.0);

        // // Compute the relative errors of the first-order optimality conditions.
        // ex = rx.array() / (1 + fx.array().abs());
    }

    /// Decompose the saddle point matrix.
    /// @note Ensure method @ref canonicalize is called first.
    auto decompose() -> void
    {
        // Decompose the saddle point matrix.
        // This decomposition is later used in method solve, possibly many
        // times! Consider lower/upper unstable variables as "fixed" variables
        // in the saddle point problem. Reason: we do not need to compute
        // Newton steps for the currently unstable variables!
        spsolver.decompose();
    }

    // /// Solve the saddle point problem.
    // /// @note Ensure method @ref decompose is called first.
    // auto solve(StepperSolveArgs args) -> void
    // {
    //     // Auxiliary constant references
    //     const auto x  = args.x;
    //     const auto p  = args.p;
    //     const auto y  = args.y;
    //     const auto z  = args.z;
    //     const auto fx = args.fx;
    //     const auto b  = args.b;
    //     const auto h  = args.h;
    //     const auto v  = args.v;
    //     const auto& stability = args.stability;

    //     // Auxiliary references
    //     auto dx = args.dx;
    //     auto dp = args.dp;
    //     auto dy = args.dy;
    //     auto dz = args.dz;

    //     // The indices of all unstable variables
    //     auto ju = stability.indicesUnstableVariables();

    //     // The indices of the strictly lower and upper unstable variables
    //     auto jsu = stability.indicesStrictlyUnstableVariables();

    //     // In the computation of xbar and ybar below use x' where x'[i] is x[i]
    //     // if i is not a strictly unstable variable, and zero if so. This is to
    //     // ensure that the strictly unstable variables are not even taken into
    //     // account in the calculation, not even in the linear equality
    //     // constraints. It is like if they were not part of the problem.

    //     auto xprime = dx; // use dx as workspace for x'
    //     xprime = x;
    //     xprime(jsu).fill(0.0);

    //     // Solve the saddle point problem.
    //     // Note: For numerical accuracy, it is important to compute
    //     // directly the next x and y iterates, instead of dx and dy.
    //     // This is because the latter causes very small values on the
    //     // right-hand side of the saddle point problem, and algebraic
    //     // manipulation of these small values results in round-off errors.
    //     spsolver.solve({ xbar, pbar, ybar, zbar });

    //     // Finalize the computation of the steps dx, dp, dy, dz
    //     dx.noalias() = xbar - xprime;
    //     dp.noalias() = pbar - p;
    //     dy.noalias() = ybar - y;
    //     dz.noalias() = zbar - z;

    //     // Replace NaN values by zero step lengths. If NaN is produced
    //     // following a saddle point solve operation, this indicates the LU
    //     // solver detected linearly dependent rows. Variables associated with
    //     // these rows are excluded from the solution procedure of the linear
    //     // system of equations. We ensure here that such variables remain
    //     // constant during the next stepping operation, by setting their step
    //     // lengths to zero.
    //     dx = dx.array().isNaN().select(0.0, dx);
    //     dy = dy.array().isNaN().select(0.0, dy);
    // }

    /// Solve the saddle point problem.
    /// @note Ensure method @ref decompose is called first.
    auto solve(StepperSolveArgs args) -> void
    {
        // Auxiliary constant references
        const auto x  = args.x;
        const auto p  = args.p;
        const auto y  = args.y;
        const auto z  = args.z;
        const auto fx = args.fx;
        const auto b  = args.b;
        const auto h  = args.h;
        const auto v  = args.v;
        const auto& stability = args.stability;

        auto dx = args.dx;
        auto dp = args.dp;
        auto dy = args.dy;
        auto dz = args.dz;

        spsolver.solve({ dx, dp, dy, dz });

        // // Replace NaN values by zero step lengths. If NaN is produced
        // // following a saddle point solve operation, this indicates the LU
        // // solver detected linearly dependent rows. Variables associated with
        // // these rows are excluded from the solution procedure of the linear
        // // system of equations. We ensure here that such variables remain
        // // constant during the next stepping operation, by setting their step
        // // lengths to zero.
        // dx = dx.array().isNaN().select(0.0, dx);
        // dp = dp.array().isNaN().select(0.0, dp);
        // dy = dy.array().isNaN().select(0.0, dy);
        // dz = dz.array().isNaN().select(0.0, dz);
    }

    /// Compute the sensitivity derivatives of the saddle point problem.
    auto sensitivities(StepperSensitivitiesArgs args) -> void
    {
        auto fxw = args.fxw;
        auto hw  = args.hw;
        auto bw  = args.bw;
        auto vw  = args.vw;
        auto xw  = args.xw;
        auto pw  = args.pw;
        auto yw  = args.yw;
        auto zw  = args.zw;
        auto sw  = args.sw;

        auto const& stability = args.stability;

        const auto nw = xw.cols();

        assert( fxw.rows() == nx );
        assert(  hw.rows() == nz );
        assert(  bw.rows() == ny );
        assert(  vw.rows() == np );
        assert(  xw.rows() == nx );
        assert(  pw.rows() == np );
        assert(  yw.rows() == ny );
        assert(  zw.rows() == nz );
        assert(  sw.rows() == nx );
        assert( fxw.cols() == nw );
        assert(  hw.cols() == nw );
        assert(  bw.cols() == nw );
        assert(  vw.cols() == nw );
        assert(  xw.cols() == nw );
        assert(  pw.cols() == nw );
        assert(  yw.cols() == nw );
        assert(  zw.cols() == nw );
        assert(  sw.cols() == nw );

        auto js = stability.indicesStableVariables();
        auto ju = stability.indicesUnstableVariables();

        xw(js, all) = -fxw(js, all);
        xw(ju, all).fill(0.0);
        pw = -vw;
        zw = -hw;
        yw =  bw;

        for(Index i = 0; i < nw; ++i)
        {
            spsolver.rhs({ xw.col(i), pw.col(i), yw.col(i), zw.col(i) });
            spsolver.solve({ xw.col(i), pw.col(i), yw.col(i), zw.col(i) });
        }

        sw(js, all).fill(0.0);
        sw(ju, all) = fxw(ju, all) + tr(Ax(all, ju)) * yw + tr(hx(all, ju)) * zw;
    }

    /// Compute the steepest descent direction with respect to Lagrange function.
    auto steepestDescentLagrange(StepperSteepestDescentLagrangeArgs args) -> void
    {
        auto [x, p, y, z, fx, b, h, v, dx, dp, dy, dz] = args;

        const auto& stability = stabilitychecker.stability();
        const auto ju = stability.indicesUnstableVariables();

        dx.noalias() = -(fx + tr(Ax)*y + tr(hx)*z);
        dx(ju).fill(0.0); // ensure zero step for unstable variables as they should continue sticked on their bounds.
        dp.noalias() = -v;
        dz.noalias() = -h;
        dy.noalias() = -(Ax*x + Ap*p - b);
    }

    /// Compute the steepest descent direction with respect to error function.
    auto steepestDescentError(StepperSteepestDescentErrorArgs args) -> void
    {
        // // Unpack the data members in args
        // auto [x, p, y, z, fx, fxx, fxp, b, h, hx, hp, v, vx, vp, dx, dp, s] = args;

        // auto dxL = xbar;
        // auto dpL = pbar;
        // auto dyL = ybar;

        // // Views to sub-vectors dylL and dynL in dyL = [dylL, dynL]
        // const auto dylL = dyL.head(ny);
        // const auto dynL = dyL.tail(mn);

        // steepestDescentLagrange({ x, p, y, z, fx, b, h, v, dxLs dpL, dyL });

        // const auto A = W.topRows(ny);

        // W.bottomRows(mn) = J;

        // dx.noalias() = tr(Ax)*dylL + tr(hx)*dynL;
        // if(options.kkt.method == SaddlePointMethod::Rangespace)
        //     dx.noalias() += H.diagonal().cwiseProduct(dxL);
        // else dx.noalias() += H * dxL;

        // dy.noalias() = A*dxL;
    }
};

Stepper::Stepper(StepperInitArgs args)
: pimpl(new Impl(args))
{}

Stepper::Stepper(const Stepper& other)
: pimpl(new Impl(*other.pimpl))
{}

Stepper::~Stepper()
{}

auto Stepper::operator=(Stepper other) -> Stepper&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto Stepper::setOptions(const Options& options) -> void
{
    pimpl->options = options;
    pimpl->spsolver.setOptions(options.kkt);
}

auto Stepper::initialize(StepperInitializeArgs args) -> void
{
    pimpl->initialize(args);
}

auto Stepper::canonicalize(StepperCanonicalizeArgs args) -> void
{
    pimpl->canonicalize(args);
}

auto Stepper::residuals(StepperResidualsArgs args) -> void
{
    pimpl->residuals(args);
}

auto Stepper::decompose() -> void
{
    pimpl->decompose();
}

auto Stepper::solve(StepperSolveArgs args) -> void
{
    pimpl->solve(args);
}

auto Stepper::sensitivities(StepperSensitivitiesArgs args) -> void
{
    pimpl->sensitivities(args);
}

auto Stepper::steepestDescentLagrange(StepperSteepestDescentLagrangeArgs args) -> void
{
    pimpl->steepestDescentLagrange(args);
}

auto Stepper::steepestDescentError(StepperSteepestDescentErrorArgs args) -> void
{
    pimpl->steepestDescentError(args);
}

} // namespace Optima
