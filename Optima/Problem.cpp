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

#include "Problem.hpp"

// Optima includes
#include <Optima/Utils.hpp>

namespace Optima {

struct Problem::Impl
{
    Matrix Aex;    ///< The coefficient matrix Aex in the linear equality constraints Aex*x + Aep*p = be.
    Matrix Aep;    ///< The coefficient matrix Aep in the linear equality constraints Aex*x + Aep*p = be.
    Matrix Agx;    ///< The coefficient matrix Agx in the linear inequality constraints Agx*x + Agp*p >= bg.
    Matrix Agp;    ///< The coefficient matrix Agp in the linear inequality constraints Agx*x + Agp*p >= bg.
    Vector be;     ///< The right-hand side vector \eq{b_{\mathrm{e}}} in the linear equality constraints Aex*x + Aep*p = be.
    Vector bg;     ///< The right-hand side vector \eq{b_{\mathrm{g}}} in the linear inequality constraints Agx*x + Agp*p >= bg.
    Vector xlower; ///< The lower bounds of the primal variables x.
    Vector xupper; ///< The upper bounds of the primal variables x.
    Vector plower; ///< The lower bounds of the paramter variables p.
    Vector pupper; ///< The upper bounds of the paramter variables p.

    /// Construct a default Problem::Impl instance.
    Impl()
    {}

    /// Construct a Problem::Impl instance with given number of variables.
    Impl(const Dims& dims)
    : Aex(zeros(dims.be, dims.x)), Aep(zeros(dims.be, dims.p)),
      Agx(zeros(dims.bg, dims.x)), Agp(zeros(dims.bg, dims.p)),
      be(zeros(dims.be)), bg(zeros(dims.bg)),
      xlower(constants(dims.x, -infinity())), xupper(constants(dims.x, infinity())),
      plower(constants(dims.p, -infinity())), pupper(constants(dims.p, infinity()))
    {}
};

Problem::Problem(const Dims& dims)
: pimpl(new Impl(dims)),
  dims(dims),
  Aex(pimpl->Aex),
  Aep(pimpl->Aep),
  Agx(pimpl->Agx),
  Agp(pimpl->Agp),
  be(pimpl->be),
  bg(pimpl->bg),
  xlower(pimpl->xlower),
  xupper(pimpl->xupper),
  plower(pimpl->plower),
  pupper(pimpl->pupper)
{}

Problem::Problem(const Problem& other)
: pimpl(new Impl(*other.pimpl)),
  dims(other.dims),
  Aex(pimpl->Aex),
  Aep(pimpl->Aep),
  Agx(pimpl->Agx),
  Agp(pimpl->Agp),
  be(pimpl->be),
  bg(pimpl->bg),
  xlower(pimpl->xlower),
  xupper(pimpl->xupper),
  plower(pimpl->plower),
  pupper(pimpl->pupper)
{}

Problem::~Problem()
{}

} // namespace Optima
