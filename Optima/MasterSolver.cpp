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
#include <Optima/TransformStep.hpp>

namespace Optima {

const auto CONTINUE = true;
const auto STOP     = false;

struct MasterSolver::Impl
{
    const MasterDims dims;
    ResidualFunction F;
    ResidualErrors E;
    MasterVector uo;
    NewtonStep newtonstep;
    TransformStep transformstep;
    ErrorControl errorcontrol;
    Convergence convergence;
    Outputter outputter; ///< The object used to output the current state of the computation.
    Result result;
    Options options;

    Impl(const MasterDims& dims)
    : dims(dims), F(dims), E(dims), uo(dims),
      newtonstep(dims),
      transformstep(dims),
      errorcontrol(dims),
      convergence()
    {
    }

    auto outputHeaderTop() -> void
    {
        if(!options.output.active) return;

        outputter.addEntry("Iteration");
        outputter.addEntry("f");
        outputter.addEntry("Error");
        outputter.addEntry("||ex||max");
        outputter.addEntry("||ep||max");
        outputter.addEntry("||ew||max");
        outputter.addEntries(options.output.xprefix, dims.nx, options.output.xnames);
        outputter.addEntries(options.output.pprefix, dims.np, options.output.pnames);
        outputter.addEntries(options.output.yprefix, dims.ny, options.output.ynames);
        outputter.addEntries(options.output.zprefix, dims.nz, options.output.znames);
        outputter.addEntries(options.output.sprefix, dims.nx, options.output.xnames);
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
        outputter.addValue(E.error);
        outputter.addValue(E.errorx);
        outputter.addValue(E.errorp);
        outputter.addValue(E.errorw);
        outputter.addValues(uo.x);
        outputter.addValues(uo.p);
        outputter.addValues(uo.w.head(dims.ny));
        outputter.addValues(uo.w.tail(dims.nz));
        outputter.addValues(Fres.stabilitystatus.s);
        outputter.addValues(E.ex);
        outputter.addValues(E.ep);
        outputter.addValues(E.ew.head(dims.ny));
        outputter.addValues(E.ew.tail(dims.nz));
        std::stringstream ss;
        if(xnames.empty()) for(auto i : jb) ss << i << " ";
        else for(auto i : jb) ss << xnames[i] << " ";
        outputter.addValue(ss.str());
        outputter.outputState();
    };

    auto solve(const MasterProblem& problem, MasterVectorRef u) -> Result
    {
        initialize(problem, u);
        while(stepping(u))
            step(u);
        finalize();
        return result;
    }

    auto setOptions(const Options& opts) -> void
    {
        options = opts;
        newtonstep.setOptions(opts.newtonstep);
        convergence.setOptions(opts.convergence);
        outputter.setOptions(opts.output);
    }

    auto initialize(const MasterProblem& problem, MasterVectorRef u) -> void
    {
        sanitycheck(problem, u);
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
        outputter.clear();
        outputHeaderTop();
    }

    auto stepping(MasterVectorRef u) -> bool
    {
        if(result.iterations > options.maxiterations)
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
            return STOP;

        return CONTINUE;
    }

    auto step(MasterVectorRef u) -> void
    {
        outputCurrentState();
        newtonstep.apply(F, uo, u);
        transformstep.execute(uo, u, F, E);
        errorcontrol.execute(uo, u, F, E);
        uo = u;
        result.iterations += 1;
    }

    auto finalize() -> void
    {
        result.succeeded = convergence.converged();
        outputCurrentState();
        outputHeaderBottom();
    }

    auto sanitycheck(const MasterProblem& problem, MasterVectorRef u) -> void
    {
        assert(problem.f.initialized());
        assert(dims.nz == 0 || problem.h.initialized());
        assert(dims.np == 0 || problem.v.initialized());
        assert(dims.nx == problem.xlower.size());
        assert(dims.nx == problem.xupper.size());
        assert(dims.np == problem.plower.size());
        assert(dims.np == problem.pupper.size());
        assert(dims.nx == u.x.size());
        assert(dims.np == u.p.size());
        assert(dims.nw == u.w.size());
    }
};

MasterSolver::MasterSolver(const MasterDims& dims)
: pimpl(new Impl(dims))
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

auto MasterSolver::solve(const MasterProblem& problem, MasterVectorRef u) -> Result
{
    return pimpl->solve(problem, u);
}

} // namespace Optima
