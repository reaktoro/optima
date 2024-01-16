// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundationither version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include "TransformStep.hpp"

// Optima includes
#include <Optima/Constants.hpp>

namespace Optima {

struct TransformStep::Impl
{
    MasterDims dims;       ///< The dimensions of the master variables.
    MasterVector ubkp;     ///< The backup master variables in case of failure.
    Vector xlower;         ///< The lower bounds for variables *x*.
    Vector xupper;         ///< The upper bounds for variables *x*.
    TransformFunction phi; ///< The custom variable transformation function.

    Impl()
    {}

    auto initialize(const MasterProblem& problem) -> void
    {
        dims = problem.dims;
        xlower = problem.xlower;
        xupper = problem.xupper;
        phi = problem.phi;
    }

    auto execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> bool
    {
        if(phi == nullptr)
            return FAILED;

        ubkp = u;

        const auto outcome = phi(uo.x, u.x);

        if(outcome == FAILED) {
            u = ubkp;
            F.update(u);
            E.update(u, F);
            return FAILED;
        }

        u.x.noalias() = min(max(u.x, xlower), xupper);

        const auto errorcurr = E.error();

        F.update(u);
        E.update(u, F);

        const auto errornext = E.error();

        if(errornext > errorcurr) {
            u = ubkp;
            F.update(u);
            E.update(u, F);
            return FAILED;
        }

        return SUCCEEDED;
    }
};

TransformStep::TransformStep()
: pimpl(new Impl())
{}

TransformStep::TransformStep(const TransformStep& other)
: pimpl(new Impl(*other.pimpl))
{}

TransformStep::~TransformStep()
{}

auto TransformStep::operator=(TransformStep other) -> TransformStep&
{
    pimpl = std::move(other.pimpl);
    return *this;
}

auto TransformStep::initialize(const MasterProblem& problem) -> void
{
    return pimpl->initialize(problem);
}

auto TransformStep::execute(MasterVectorView uo, MasterVectorRef u, ResidualFunction& F, ResidualErrors& E) -> bool
{
    return pimpl->execute(uo, u, F, E);
}

} // namespace Optima
