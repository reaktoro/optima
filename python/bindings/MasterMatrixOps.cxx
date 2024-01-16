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
#include <Optima/MasterMatrixOps.hpp>
using namespace Optima;

void exportMasterMatrixOps(py::module& m)
{
    py::class_<MasterMatrixTrExpr>(m, "MasterMatrixTrExpr")
        .def_property_readonly("M", [](const MasterMatrixTrExpr& self) { return self.M; })
        .def("__mul__", [](const MasterMatrixTrExpr& l, const MasterVectorView& r) { return l * r; })
        ;

    m.def("tr", [](const MasterMatrix& M) { return tr(M); });
}
