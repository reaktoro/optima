// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2023 Allan Leal
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

#include "MasterSolver.hpp"

// Optima includes
#include <Optima/Convergence.hpp>
#include <Optima/ErrorControl.hpp>
#include <Optima/Exception.hpp>
#include <Optima/MasterProblem.hpp>
#include <Optima/MasterVector.hpp>
#include <Optima/NewtonStep.hpp>
#include <Optima/Options.hpp>
#include <Optima/Outputter.hpp>
#include <Optima/ResidualErrors.hpp>
#include <Optima/ResidualFunction.hpp>
#include <Optima/Result.hpp>
#include <Optima/SensitivitySolver.hpp>
#include <Optima/TransformStep.hpp>

namespace Optima {

const auto CONTINUE = true;
const auto STOP     = false;

struct MasterSolver::Impl
{
    MasterDims dims;
    ResidualFunction F;
    ResidualErrors E;
    MasterVector uo;
    NewtonStep newtonstep;
    TransformStep transformstep;
    ErrorControl errorcontrol;
    Convergence convergence;
    SensitivitySolver sensitivitysolver;
    Outputter outputter; ///< The object used to output the current state of the computation.
    Result result;
    Options options;

    Impl()
    {}

    auto outputHeaderTop() -> void
    {
        if(!options.output.active) return;

        outputter.addEntry("Iteration");
        outputter.addEntry("f");
        outputter.addEntry("Error");
        outputter.addEntry("||ex||max");
        outputter.addEntry("||ep||max");
        outputter.addEntry("||ew||max");
        outputter.addEntries("x", dims.nx, options.output.xnames);
        outputter.addEntries("p", dims.np, options.output.pnames);
        outputter.addEntries("y", dims.ny, options.output.ynames);
        outputter.addEntries("z", dims.nz, options.output.znames);
        outputter.addEntries("s", dims.nx, options.output.xnames);
        outputter.addEntries("ex", dims.nx, options.output.xnames);
        outputter.addEntries("ep", dims.np, options.output.pnames);
        outputter.addEntries("ey", dims.ny, options.output.ynames);
        outputter.addEntries("ez", dims.nz, options.output.znames);
        outputter.addEntry("Basic Variables");
        outputter.outputHeader();
    };

    auto outputHeaderBottom() -> void
    {
        if(!options.output.active) return;
        outputter.outputHeader();
    };

    auto outputCurrentState() -> void
    {
        if(!options.output.active) return;
        const auto& Fres = F.result();
        const auto& jb = Fres.Jm.RWQ.jb;
        const auto& xnames = options.output.xnames;
        outputter.addValue(result.iterations);
        outputter.addValue(Fres.f.f);
        outputter.addValue(E.error());
        outputter.addValue(E.errorx());
        outputter.addValue(E.errorp());
        outputter.addValue(E.errorw());
        outputter.addValues(uo.x);
        outputter.addValues(uo.p);
        outputter.addValues(uo.w.head(dims.ny));
        outputter.addValues(uo.w.tail(dims.nz));
        outputter.addValues(Fres.stabilitystatus.s);
        outputter.addValues(E.ex());
        outputter.addValues(E.ep());
        outputter.addValues(E.ew().head(dims.ny));
        outputter.addValues(E.ew().tail(dims.nz));
        std::stringstream ss;
        if(xnames.empty()) for(auto i : jb) ss << i << " ";
        else for(auto i : jb) ss << xnames[i] << " ";
        outputter.addValue(ss.str());
        outputter.outputState();
    };

    auto solve(const MasterProblem& problem, MasterState& state) -> Result
    {
        auto& u = state.u;
        initialize(problem, u);
        while(stepping(u))
            step(u);
        finalize(state);
        return result;
    }

    auto solve(const MasterProblem& problem, MasterState& state, MasterSensitivity& sensitity) -> Result
    {
        solve(problem, state);
        F.updateOnlyJacobian(state.u); // update the Jacobian matrices wrt x, p, c
        sensitivitysolver.solve(F, state, sensitity);
        return result;
    }

    auto setOptions(const Options& opts) -> void
    {
        options = opts;
        newtonstep.setOptions(opts.newtonstep);
        convergence.setOptions(opts.convergence);
        errorcontrol.setOptions({opts.errorstatus, opts.backtracksearch, opts.linesearch});
        outputter.setOptions(opts.output);
    }

    auto initialize(const MasterProblem& problem, MasterVectorRef u) -> void
    {
        sanitycheck(problem, u);
        dims = problem.dims;
        result = {};
        u.x.noalias() = min(max(u.x, problem.xlower), problem.xupper);
        u.p.noalias() = min(max(u.p, problem.plower), problem.pupper);
        uo = u;
        F.initialize(problem);
        E.initialize(problem);
        transformstep.initialize(problem);
        newtonstep.initialize(problem);
        errorcontrol.initialize(problem);
        convergence.initialize(problem);
        sensitivitysolver.initialize(problem);
        outputter.clear();
        outputHeaderTop();
    }

    auto stepping(MasterVectorRef u) -> bool
    {
        if(result.iterations > options.maxiters)
            return STOP;

        // At the beginning of each new iteration, evaluate the residual
        // function and the error. Stop if the error is already low enough.
        // Note that this always produce a final state with update residual
        // function and its derivatives should they be needed for calculation
        // of the sensitity derivatives of the solution.

        F.update(u);
        E.update(u, F);
        convergence.update(E);

        if(convergence.converged())
            if(result.iterations > 0 || !options.convergence.requires_at_least_one_iteration)
                return STOP;

        return CONTINUE;
    }

    auto step(MasterVectorRef u) -> void
    {
        outputCurrentState();
        newtonstep.apply(F, uo, u);
        transformstep.execute(uo, u, F, E); // TODO: Check if E should not be altered here, which could compromise errorcontrol logic!
        errorcontrol.execute(uo, u, F, E);
        uo = u;
        result.iterations += 1;
    }

    auto finalize(MasterState& state) -> void
    {
        result.succeeded = convergence.converged();
        outputCurrentState();
        outputHeaderBottom();
        auto const& Fresult = F.result();
        auto const& ss = Fresult.stabilitystatus;
        state.s = ss.s;
        state.js = ss.js;
        state.ju = ss.ju;
        state.jlu = ss.jlu;
        state.juu = ss.juu;
        state.jb = Fresult.Jc.jb;
        state.jn = Fresult.Jc.jn;
    }

    auto sanitycheck(const MasterProblem& problem, MasterVectorRef u) -> void
    {
        assert(problem.f.initialized());
        assert(problem.dims.nz == 0 || problem.h.initialized());
        assert(problem.dims.np == 0 || problem.v.initialized());
        assert(problem.dims.nx == problem.xlower.size());
        assert(problem.dims.nx == problem.xupper.size());
        assert(problem.dims.np == problem.plower.size());
        assert(problem.dims.np == problem.pupper.size());
        assert(problem.dims.nx == u.x.size());
        assert(problem.dims.np == u.p.size());
        assert(problem.dims.nw == u.w.size());
    }
};

MasterSolver::MasterSolver()
: pimpl(new Impl())
{}

MasterSolver::MasterSolver(const MasterSolver& other)
: pimpl(new Impl(*other.pimpl))
{}

MasterSolver::~MasterSolver()
{}

auto MasterSolver::operator=(MasterSolver other) -> MasterSolver&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto MasterSolver::setOptions(const Options& options) -> void
{
    pimpl->setOptions(options);
}

auto MasterSolver::solve(const MasterProblem& problem, MasterState& state) -> Result
{
    return pimpl->solve(problem, state);
}

auto MasterSolver::solve(const MasterProblem& problem, MasterState& state, MasterSensitivity& sensitivity) -> Result
{
    return pimpl->solve(problem, state, sensitivity);
}

} // namespace Optima
