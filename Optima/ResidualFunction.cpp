// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2020 Allan Leal
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
#include <Optima/Canonicalizer.hpp>
#include <Optima/EchelonizerW.hpp>
#include <Optima/Exception.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/ResidualVector.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct ResidualFunction::Impl
{
    /// The dimensions of the master variables.
    MasterDims dims;

    /// The result of the evaluation of f(x, p).
    ObjectiveResult fres;

    /// The result of the evaluation of h(x, p).
    ConstraintResult hres;

    /// The result of the evaluation of v(x, p).
    ConstraintResult vres;

    /// The echelonizer of matrix W = [Wx Wp] = [Ax Ap; Jx Jp].
    EchelonizerW echelonizerW;

    /// The priority weights for selection of basic variables in x.
    Vector wx;

    /// The current stability status of the x variables.
    Stability stability;

    /// The canonicalizer of the Jacobian matrix of the residual function.
    Canonicalizer canonicalizer;

    /// The current state of the residual vector.
    ResidualVector residual;

    /// The objective function *f(x, p)*.
    ObjectiveFunction f;

    /// The nonlinear equality constraint function *h(x, p)*.
    ConstraintFunction h;

    /// The external nonlinear constraint function *v(x, p)*.
    ConstraintFunction v;

    /// The right-hand side vector b in the linear equality constraints.
    Vector b;

    /// The lower bounds for variables *x*.
    Vector xlower;

    /// The upper bounds for variables *x*.
    Vector xupper;

    /// The vector with the sensitive parameters *c*.
    Vector c;

    /// True if the last update call succeeded.
    bool succeeded = false;

    Impl()
    {}

    auto initialize(const MasterProblem& problem) -> void
    {
        const auto nx = problem.dims.nx;
        const auto np = problem.dims.np;
        const auto ny = problem.dims.ny;
        const auto nz = problem.dims.nz;
        const auto nc = problem.c.size();
        dims = problem.dims;
        fres.resize(nx, np, nc);
        hres.resize(nz, nx, np, nc);
        vres.resize(np, nx, np, nc);
        echelonizerW.initialize(dims, problem.Ax, problem.Ap);
        f = problem.f;
        h = problem.h;
        v = problem.v;
        b = problem.b;
        xlower = problem.xlower;
        xupper = problem.xupper;
        c = problem.c;
    }

    auto update(MasterVectorView u) -> void
    {
        sanitycheck(u);
        succeeded = updateFunctionEvals(u); // currently, even if succeeded==false, let the remaining lines be executed, otherwise result() fails (at least in Windows).
        updateEchelonFormMatrixW(u);
        updateIndicesStableVariables(u);
        updateCanonicalFormJacobianMatrix(u);
        updateResidualVector(u);
    }

    auto updateSkipJacobian(MasterVectorView u) -> void
    {
        sanitycheck(u);
        succeeded = updateFunctionEvalsSkippingJacobianEvals(u);
        updateEchelonFormMatrixW(u);
        updateIndicesStableVariables(u);
        updateCanonicalFormJacobianMatrix(u);
        updateResidualVector(u);
    }

    auto updateFunctionEvalsAux(MasterVectorView u, bool eval_ddx, bool eval_ddp, bool eval_ddc) -> bool
    {
        const auto x = u.x;
        const auto p = u.p;
        const auto RWQ = echelonizerW.RWQ();
        const auto ibasicvars = RWQ.jb;
        ObjectiveOptions  fopts{{eval_ddx, eval_ddp && dims.np, eval_ddc}, ibasicvars};
        ConstraintOptions hopts{{eval_ddx, eval_ddp && dims.np, eval_ddc}, ibasicvars};
        ConstraintOptions vopts{{eval_ddx, eval_ddp && dims.np, eval_ddc}, ibasicvars};
        f(fres, x, p, c, fopts);
        h(hres, x, p, c, hopts);
        v(vres, x, p, c, vopts);
        return succeeded = fres.succeeded && hres.succeeded && vres.succeeded;
    }

    auto updateFunctionEvals(MasterVectorView u) -> bool
    {
        bool eval_ddx = true, eval_ddp = true, eval_ddc = false;
        return updateFunctionEvalsAux(u, eval_ddx, eval_ddp, eval_ddc);
    }

    auto updateFunctionEvalsSkippingJacobianEvals(MasterVectorView u) -> bool
    {
        bool eval_ddx = false, eval_ddp = false, eval_ddc = false;
        return updateFunctionEvalsAux(u, eval_ddx, eval_ddp, eval_ddc);
    }

    auto updateEchelonFormMatrixW(MasterVectorView u) -> void
    {
        const auto& x = u.x;
        const auto& Jx = hres.ddx;
        const auto& Jp = hres.ddp;

        wx.noalias() = abs(x);
        wx.noalias() = (x.array() != xlower.array()).select(wx, -1.0); // Enforce weak priority for variables on the bounds.
        wx.noalias() = (x.array() != xupper.array()).select(wx, -1.0); // Enforce weak priority for variables on the bounds.
        echelonizerW.update(Jx, Jp, wx);
    }

    auto updateIndicesStableVariables(MasterVectorView u) -> void
    {
        const auto& fx = fres.fx;
        const auto& x = u.x;
        const auto& w = u.w;
        const auto& Wx = echelonizerW.W().Wx;
        const auto& jb = echelonizerW.RWQ().jb;
        stability.update({Wx, fx, x, w, xlower, xupper, jb});
    }

    auto updateCanonicalFormJacobianMatrix(MasterVectorView u) -> void
    {
        canonicalizer.update(jacobianMatrixMasterForm());
    }

    auto updateResidualVector(MasterVectorView u) -> void
    {
        const auto& fx = fres.fx;
        const auto& h = hres.val;
        const auto& v = vres.val;
        const auto& W = echelonizerW.W();
        const auto& Wx = W.Wx;
        const auto& Wp = W.Wp;
        const auto& x = u.x;
        const auto& p = u.p;
        const auto& w = u.w;
        const auto& y = w.head(dims.ny);
        const auto& z = w.tail(dims.nz);
        const auto& Jc = jacobianMatrixCanonicalForm();
        residual.update({Jc, Wx, Wp, x, p, y, z, fx, v, b, h});
    }

    auto jacobianMatrixMasterForm() const -> MasterMatrix
    {
        const auto& stabilitystatus = stability.status();
        const auto& js = stabilitystatus.js;
        const auto& ju = stabilitystatus.ju;
        const auto& H = MatrixViewH{fres.fxx, fres.fxp, fres.diagfxx};
        const auto& V = MatrixViewV{vres.ddx, vres.ddp};
        const auto& W = echelonizerW.W();
        const auto& RWQ = echelonizerW.RWQ();
        return {dims, H, V, W, RWQ, js, ju};
    }

    auto jacobianMatrixCanonicalForm() const -> CanonicalMatrix
    {
        return canonicalizer.canonicalMatrix();
    }

    auto residualVectorMasterForm() const -> MasterVectorView
    {
        return residual.masterVector();
    }

    auto residualVectorCanonicalForm() const -> CanonicalVectorView
    {
        return residual.canonicalVector();
    }

    auto result() const -> ResidualFunctionResult
    {
        const auto Jm = jacobianMatrixMasterForm();
        const auto Jc = jacobianMatrixCanonicalForm();
        const auto Fm = residualVectorMasterForm();
        const auto Fc = residualVectorCanonicalForm();
        const auto stabilitystatus = stability.status();

        return { fres, hres, vres, Jm, Jc, Fm, Fc, stabilitystatus, succeeded };
    }

    auto sanitycheck(MasterVectorView u) const -> void
    {
        assert(b.size() == dims.ny);
        assert(xlower.size() == dims.nx);
        assert(xupper.size() == dims.nx);
        assert(u.x.size() == dims.nx);
        assert(u.p.size() == dims.np);
        assert(u.w.size() == dims.nw);
    }
};

ResidualFunction::ResidualFunction()
: pimpl(new Impl())
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
    return pimpl->initialize(problem);
}

auto ResidualFunction::update(MasterVectorView u) -> void
{
    pimpl->update(u);
}

auto ResidualFunction::updateSkipJacobian(MasterVectorView u) -> void
{
    pimpl->updateSkipJacobian(u);
}

auto ResidualFunction::result() const -> ResidualFunctionResult
{
    return pimpl->result();
}

} // namespace Optima
