// Optima is a C++ library for solving linear and non-linear constrained optimization problems.
//
// Copyright © 2020-2024 Allan Leal
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

// pybind11 includes
#include "pybind11.hxx"

// Optima includes
#include <Optima/Options.hpp>
using namespace Optima;

void exportOptions(py::module& m)
{
    py::class_<OutputOptions, OutputterOptions>(m, "OutputOptions")
        .def(py::init<>())
        .def_readwrite("xnames"   , &OutputOptions::xnames)
        .def_readwrite("xbgnames" , &OutputOptions::xbgnames)
        .def_readwrite("xhgnames" , &OutputOptions::xhgnames)
        .def_readwrite("pnames"   , &OutputOptions::pnames)
        .def_readwrite("ynames"   , &OutputOptions::ynames)
        .def_readwrite("znames"   , &OutputOptions::znames)
        ;

    py::class_<SteepestDescentOptions>(m, "SteepestDescentOptions")
        .def_readwrite("tolerance", &SteepestDescentOptions::tolerance)
        .def_readwrite("maxiters", &SteepestDescentOptions::maxiters)
        ;

    py::class_<Options>(m, "Options")
        .def(py::init<>())
        .def_readwrite("output"         , &Options::output         , "The options for the output of the optimization calculations")
        .def_readwrite("maxiters"       , &Options::maxiters       , "The maximum number of iterations in the optimization calculations.")
        .def_readwrite("errorstatus"    , &Options::errorstatus    , "The options for assessing error status.")
        .def_readwrite("backtracksearch", &Options::backtracksearch, "The options for the backtrack search operation.")
        .def_readwrite("linesearch"     , &Options::linesearch     , "The options for the linear search minimization operation.")
        .def_readwrite("steepestdescent", &Options::steepestdescent, "The options for the steepest descent step operation when needed.")
        .def_readwrite("newtonstep"     , &Options::newtonstep     , "The options used for Newton step calculations.")
        .def_readwrite("convergence"    , &Options::convergence    , "The options used for convergence analysis.")
        ;
}
