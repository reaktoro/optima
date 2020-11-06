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

#include "ResidualFunction.hpp"

// Optima includes
#include <Optima/CanonicalMatrix.hpp>
#include <Optima/Exception.hpp>
#include <Optima/ResidualVector.hpp>
#include <Optima/Result.hpp>
#include <Optima/Stability2.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct ResidualFunction::Impl
{
    const MasterDims dims; ///< The dimensions of the master variables.
    const Matrix Ax;       ///< The coefficient matrix Ax of the linear equality constraints.
    const Matrix Ap;       ///< The coefficient matrix Ap of the linear equality constraints.

    ObjectiveFunction evalf;  ///< The objective function f(x, p).
    ConstraintFunction evalh; ///< The nonlinear equality constraint function h(x, p).
    ConstraintFunction evalv; ///< The external nonlinear constraint function v(x, p).
    Vector b;                 ///< The right-hand side vector b in the linear equality constraints.
    Vector xlower;            ///< The lower bounds for variables x.
    Vector xupper;            ///< The upper bounds for variables x.

    MatrixRWQ RWQ; ///< The echelon form of matrix W = [Wx Wp] = [Ax Ap; Jx Jp].

    double f;     ///< The evaluated value of f(x, p).
    Vector fx;    ///< The evaluated gradient of f(x, p) with respect to x.
    Matrix fxx;   ///< The evaluated Jacobian of fx(x, p) with respect to x.
    Matrix fxp;   ///< The evaluated Jacobian of fx(x, p) with respect to p.
    bool diagfxx; ///< The flag indicating whether `fxx` is diagonal.

    Vector v;   ///< The evaluated value of v(x, p).
    Matrix vx;  ///< The evaluated Jacobian of v(x, p) with respect to x.
    Matrix vp;  ///< The evaluated Jacobian of v(x, p) with respect to p.

    Vector h;   ///< The evaluated value of h(x, p).
    Matrix hx;  ///< The evaluated Jacobian of h(x, p) with respect to x.
    Matrix hp;  ///< The evaluated Jacobian of h(x, p) with respect to p.

    Vector wx;  ///< The priority weights for selection of basic variables in x.

    Vector ex;    ///< The errors associated with variables x.
    Vector ewbs;  ///< The errors associated with linear and nonlinear constraints in canonical form.

    Stability2 stability;     ///< The stability checker of variables in x.
    CanonicalMatrix jacobian; ///< The Jacobian matrix of the residual function in canonical form.
    ResidualVector residual;  ///< The object that calculates the residual vector

    Impl(const MasterProblem& problem)
    : dims(problem.dims), Ax(problem.Ax), Ap(problem.Ap),
      RWQ(dims, Ax, Ap), stability(dims.nx),
      jacobian(dims), residual(dims)
    {
        initialize(problem);

        const auto [nx, np, ny, nz, nw, nt] = dims;

        fx.resize(nx);
        fxx.resize(nx, nx);
        fxp.resize(nx, np);

        v.resize(np);
        vx.resize(np, nx);
        vp.resize(np, np);

        h.resize(nz);
        hx.resize(nz, nx);
        hp.resize(nz, np);

        wx.resize(nx);

        ex.resize(nx);
        ewbs.resize(nw);
    }

    auto initialize(const MasterProblem& problem) -> void
    {
        b      = problem.b;
        evalf  = problem.f;
        evalh  = problem.h;
        evalv  = problem.v;
        xlower = problem.xlower;
        xupper = problem.xupper;
        sanitycheck();
    }

    auto update(MasterVectorView u) -> ResidualFunctionUpdateStatus
    {
        sanitycheck();
        const auto status = updateFunctionEvaluations(u);
        if(status == FAILED)
            return status;
        updateMatrixRWQ(u);
        updateStabilityStatus(u);
        updateJacobianMatrix(u);
        updateResidualVector(u);
        return status;
    }

    auto updateSkipJacobian(MasterVectorView u) -> ResidualFunctionUpdateStatus
    {
        sanitycheck();
        const auto status = updateFunctionEvaluationsSkipJacobian(u);
        if(status == FAILED)
            return status;
        updateMatrixRWQ(u);
        updateStabilityStatus(u);
        updateJacobianMatrix(u); // needed in case the set of stable/unstable variables changed
        updateResidualVector(u);
        return status;
    }

    auto updateFunctionEvaluations(MasterVectorView u) -> ResidualFunctionUpdateStatus
    {
        const auto x = u.x;
        const auto p = u.p;
        ResidualFunctionUpdateStatus status;
        status.f = evalf(x, p, {f, fx, fxx, fxp, diagfxx}); if(status.f == FAILED) return status;
        status.h = evalh(x, p, {h, hx, hp});                if(status.h == FAILED) return status;
        status.v = evalv(x, p, {v, vx, vp});                if(status.v == FAILED) return status;
        return status;
    }

    auto updateFunctionEvaluationsSkipJacobian(MasterVectorView u) -> ResidualFunctionUpdateStatus
    {
        const auto x = u.x;
        const auto p = u.p;
        Matrix O;
        ResidualFunctionUpdateStatus status;
        status.f = evalf(x, p, {f, fx, O, O, diagfxx}); if(status.f == FAILED) return status;
        status.h = evalh(x, p, {h, O, O});              if(status.h == FAILED) return status;
        status.v = evalv(x, p, {v, O, O});              if(status.v == FAILED) return status;
        return status;
    }

    auto updateMatrixRWQ(MasterVectorView u) -> void
    {
        const auto x = u.x;
        wx = min(x - xlower, xupper - x);
        wx = wx.array().isInf().select(abs(x), wx); // replace wx[i]=inf by wx[i]=abs(x[i])
        assert(wx.minCoeff() >= 0.0);
        wx = (wx.array() > 0.0).select(wx, -1.0); // set negative priority weights for variables on the bounds
        RWQ.update(hx, hp, wx);
    }

    auto updateStabilityStatus(MasterVectorView u) -> void
    {
        stability.update({RWQ, fx, u.x, xlower, xupper});
    }

    auto updateJacobianMatrix(MasterVectorView u) -> void
    {
        jacobian.update(masterJacobianMatrix());
    }

    auto updateResidualVector(MasterVectorView u) -> void
    {
        const auto W = RWQ.asMatrixViewW();
        const auto Wx = W.Wx;
        const auto Wp = W.Wp;
        const auto x = u.x;
        const auto p = u.p;
        const auto w = u.w;
        const auto y = w.head(dims.ny);
        const auto z = w.tail(dims.nz);
        residual.update({jacobian, Wx, Wp, x, p, y, z, fx, v, b, h});
    }

    auto canonicalJacobianMatrix() const -> CanonicalMatrixView
    {
        return jacobian;
    }

    auto canonicalResidualVector() const -> CanonicalVectorView
    {
        return residual.canonicalVector();
    }

    auto masterJacobianMatrix() const -> MasterMatrix
    {
        const auto stabilitystatus = stability.status();
        const auto js = stabilitystatus.js;
        const auto ju = stabilitystatus.ju;
        const auto H = MatrixViewH{fxx, fxp, diagfxx};
        const auto V = MatrixViewV{vx, vp};
        return {H, V, RWQ, js, ju};
    }

    auto masterResidualVector() const -> MasterVectorView
    {
        return residual.masterVector();
    }

    auto sanitycheck() const -> void
    {
        assert(b.size() == dims.ny);
        assert(evalf != nullptr);
        assert(evalh != nullptr);
        assert(evalv != nullptr);
        assert(xlower.size() == dims.nx);
        assert(xupper.size() == dims.nx);
    }
};

ResidualFunction::ResidualFunction(const MasterProblem& problem)
: pimpl(new Impl(problem))
{}

ResidualFunction::ResidualFunction(const ResidualFunction& other)
: pimpl(new Impl(*other.pimpl))
{}

ResidualFunction::~ResidualFunction()
{}

auto ResidualFunction::operator=(ResidualFunction other) -> ResidualFunction&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto ResidualFunction::initialize(const MasterProblem& problem) -> void
{
    pimpl->initialize(problem);
}

auto ResidualFunction::update(MasterVectorView u) -> ResidualFunctionUpdateStatus
{
    return pimpl->update(u);
}

auto ResidualFunction::updateSkipJacobian(MasterVectorView u) -> ResidualFunctionUpdateStatus
{
    return pimpl->updateSkipJacobian(u);
}

auto ResidualFunction::canonicalJacobianMatrix() const -> CanonicalMatrixView
{
    return pimpl->canonicalJacobianMatrix();
}

auto ResidualFunction::canonicalResidualVector() const -> CanonicalVectorView
{
    return pimpl->canonicalResidualVector();
}

auto ResidualFunction::masterJacobianMatrix() const -> MasterMatrix
{
    return pimpl->masterJacobianMatrix();
}

auto ResidualFunction::masterResidualVector() const -> MasterVectorView
{
    return pimpl->masterResidualVector();
}

} // namespace Optima
