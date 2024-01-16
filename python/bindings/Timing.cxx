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
#include <Optima/Timing.hpp>
using namespace Optima;

void exportTiming(py::module& m)
{
	py::class_<Timer>(m, "Timer")
		.def(py::init<>())
		.def("elapsed", &Timer::elapsed)
		;

    m.def("timenow", &timenow);
    m.def("elapsed", (double(*)(const Time&, const Time&)) &elapsed);
    m.def("elapsed", (double(*)(const Time&)) &elapsed);
}
