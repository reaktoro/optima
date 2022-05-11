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

// pybind11 includes
#include "pybind11.hxx"

// Optima includes
#include <Optima/EchelonizerW.hpp>
using namespace Optima;

void exportEchelonizerW(py::module& m)
{
    auto initialize = [](EchelonizerW& self, const MasterDims& dims, MatrixView4py Ax, MatrixView4py Ap)
    {
        self.initialize(dims, Ax, Ap);
    };

    auto update = [](EchelonizerW& self, MatrixView4py Jx, MatrixView4py Jp, VectorView weights)
    {
        self.update(Jx, Jp, weights);
    };

    py::class_<EchelonizerW>(m, "EchelonizerW")
        .def(py::init<>())
        .def("initialize", initialize)
        .def("update", update)
        .def("W", &EchelonizerW::W, PYBIND_ENSURE_MUTUAL_EXISTENCE)
        .def("RWQ", &EchelonizerW::RWQ, PYBIND_ENSURE_MUTUAL_EXISTENCE)
        ;
}
