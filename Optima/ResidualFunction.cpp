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
#include <Optima/Canonicalizer.hpp>
#include <Optima/EchelonizerW.hpp>
#include <Optima/Exception.hpp>
#include <Optima/ResidualVector.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>

namespace Optima {

struct ResidualFunction::Impl
{
    /// The dimensions of the master variables.
    const MasterDims dims;

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
    Stability2 stability;

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

    /// True if the last update call succeeded.
    bool succeeded = false;

    Impl(const MasterDims& dims)
    : dims(dims),
      fres(dims.nx, dims.np),
      hres(dims.nz, dims.nx, dims.np),
      vres(dims.np, dims.nx, dims.np),
      echelonizerW(dims), stability(dims.nx),
      canonicalizer(dims), residual(dims)
    {
        wx.resize(dims.nx);
    }

    auto initialize(const MasterProblem& problem) -> void
    {
        echelonizerW.initialize(problem.Ax, problem.Ap);
        f      = problem.f;
        h      = problem.h;
        v      = problem.v;
        b      = problem.b;
        xlower = problem.xlower;
        xupper = problem.xupper;
    }

    auto update(MasterVectorView u) -> void
    {
        sanitycheck(u);
        const auto status = updateFunctionEvals(u);
        if(status == FAILED)
            return;
        updateEchelonFormMatrixW(u);
        updateIndicesStableVariables(u);
        updateCanonicalFormJacobianMatrix(u);
        updateResidualVector(u);
    }

    auto updateSkipJacobian(MasterVectorView u) -> void
    {
        sanitycheck(u);
        const auto status = updateFunctionEvalsSkippingJacobianEvals(u);
        if(status == FAILED)
            return;
        updateEchelonFormMatrixW(u);
        updateIndicesStableVariables(u);
        updateCanonicalFormJacobianMatrix(u);
        updateResidualVector(u);
    }

    template<bool evaljac>
    auto updateFunctionEvalsAux(MasterVectorView u) -> bool
    {
        const auto x = u.x;
        const auto p = u.p;
        const auto RWQ = echelonizerW.RWQ();
        const auto ibasicvars = RWQ.jb;
        ObjectiveOptions  fopts{{evaljac, evaljac}, ibasicvars};
        ConstraintOptions hopts{{evaljac, evaljac}, ibasicvars};
        ConstraintOptions vopts{{evaljac, evaljac}, ibasicvars};
        f(fres, x, p, fopts);
        h(hres, x, p, hopts);
        v(vres, x, p, vopts);
        return succeeded = fres.succeeded && hres.succeeded && vres.succeeded;
    }

    auto updateFunctionEvals(MasterVectorView u) -> bool
    {
        return updateFunctionEvalsAux<true>(u);
    }

    auto updateFunctionEvalsSkippingJacobianEvals(MasterVectorView u) -> bool
    {
        return updateFunctionEvalsAux<false>(u);
    }

    auto updateEchelonFormMatrixW(MasterVectorView u) -> void
    {
        const auto& x = u.x;
        const auto& Jx = hres.ddx;
        const auto& Jp = hres.ddp;
        wx = min(x - xlower, xupper - x);
        wx = wx.array().isInf().select(abs(x), wx); // replace wx[i]=inf by wx[i]=abs(x[i])
        assert(wx.minCoeff() >= 0.0);
        wx = (wx.array() > 0.0).select(wx, -1.0); // set negative priority weights for variables on the bounds
        echelonizerW.update(Jx, Jp, wx);
    }

    auto updateIndicesStableVariables(MasterVectorView u) -> void
    {
        const auto& fx = fres.fx;
        const auto& x = u.x;
        const auto RWQ = echelonizerW.RWQ();
        stability.update({RWQ, fx, x, xlower, xupper});
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

ResidualFunction::ResidualFunction(const MasterDims& dims)
: pimpl(new Impl(dims))
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
