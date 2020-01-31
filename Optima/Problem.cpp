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
    /// The coefficient matrix \eq{A_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    Matrix Ae;

    /// The coefficient matrix \eq{A_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    Matrix Ag;

    /// The right-hand side vector \eq{b_{\mathrm{e}}} in the linear equality constraints \eq{A_{\mathrm{e}}x=b_{\mathrm{e}}}.
    Vector be;

    /// The right-hand side vector \eq{b_{\mathrm{g}}} in the linear inequality constraints \eq{A_{\mathrm{g}}x\ge b_{\mathrm{g}}}.
    Vector bg;

    /// The lower bounds of the variables \eq{x}.
    Vector xlower;

    /// The upper bounds of the variables \eq{x}.
    Vector xupper;

    /// Construct a default Problem::Impl instance.
    Impl()
    {}

    /// Construct a Problem::Impl instance with given number of variables.
    Impl(const Dims& dims)
    : Ae(zeros(dims.be, dims.x)),
      Ag(zeros(dims.bg, dims.x)),
      be(zeros(dims.be)),
      bg(zeros(dims.bg)),
      xlower(constants(dims.x, -infinity())),
      xupper(constants(dims.x, infinity()))
    {}
};

Problem::Problem(const Dims& dims)
: pimpl(new Impl(dims)),
  dims(dims),
  Ae(pimpl->Ae),
  Ag(pimpl->Ag),
  be(pimpl->be),
  bg(pimpl->bg),
  xlower(pimpl->xlower),
  xupper(pimpl->xupper)
{}

Problem::Problem(const Problem& other)
: pimpl(new Impl(*other.pimpl)),
  dims(other.dims),
  Ae(pimpl->Ae),
  Ag(pimpl->Ag),
  be(pimpl->be),
  bg(pimpl->bg),
  xlower(pimpl->xlower),
  xupper(pimpl->xupper)
{}

Problem::~Problem()
{}

} // namespace Optima
