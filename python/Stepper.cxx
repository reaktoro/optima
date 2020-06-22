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

// pybind11 includes
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

// Optima includes
#include <Optima/Stepper.hpp>
#include <Optima/Options.hpp>
using namespace Optima;

void exportStepper(py::module& m)
{
    auto init = [](Index n, Index m, MatrixConstRef4py A) -> Stepper
    {
        return Stepper({n, m, A});
    };

    auto initialize = [](Stepper& self,
        VectorConstRef b,
        VectorConstRef xlower,
        VectorConstRef xupper,
        VectorRef x,
        Stability& stability)
    {
        self.initialize({ b, xlower, xupper, x, stability });
    };

    auto canonicalize = [](Stepper& self,
        VectorConstRef x,
        VectorConstRef y,
        VectorConstRef g,
        MatrixConstRef4py H,
        MatrixConstRef4py J,
        VectorConstRef xlower,
        VectorConstRef xupper,
        Stability& stability)
    {
        self.canonicalize({ x, y, g, H, J, xlower, xupper, stability  });
    };

    auto residuals = [](Stepper& self,
        VectorConstRef x,
        VectorConstRef y,
        VectorConstRef b,
        VectorConstRef h,
        VectorConstRef g,
        VectorRef rx,
        VectorRef ry,
        VectorRef ex,
        VectorRef ey,
        VectorRef z)
    {
        self.residuals({ x, y, b, h, g, rx, ry, ex, ey, z });
    };

    auto decompose = [](Stepper& self,
        VectorConstRef x,
        VectorConstRef y,
        VectorConstRef g,
        MatrixConstRef4py H,
        MatrixConstRef4py J,
        VectorConstRef xlower,
        VectorConstRef xupper,
        Stability& stability)
    {
        self.decompose({ x, y, g, H, J, xlower, xupper, stability });
    };

    auto solve = [](Stepper& self,
        VectorConstRef x,
        VectorConstRef y,
        VectorConstRef b,
        VectorConstRef h,
        VectorConstRef g,
        MatrixConstRef4py H,
        Stability const& stability,
        VectorRef dx,
        VectorRef dy)
    {
        self.solve({x, y, b, h, g, H, stability, dx, dy });
    };

    Matrix tmp_dxdp, tmp_dydp, tmp_dzdp;
    auto sensitivities = [=](Stepper& self,
        MatrixConstRef4py dgdp,
        MatrixConstRef4py dhdp,
        MatrixConstRef4py dbdp,
        Stability const& stability,
        MatrixRef4py dxdp,
        MatrixRef4py dydp,
        MatrixRef4py dzdp) mutable
    {
        tmp_dxdp.resize(dxdp.rows(), dxdp.cols());
        tmp_dydp.resize(dydp.rows(), dydp.cols());
        tmp_dzdp.resize(dzdp.rows(), dzdp.cols());
        self.sensitivities({dgdp, dhdp, dbdp, stability, tmp_dxdp, tmp_dydp, tmp_dzdp});
        dxdp = tmp_dxdp;
        dydp = tmp_dydp;
        dzdp = tmp_dzdp;
    };

    py::class_<Stepper>(m, "Stepper")
        .def(py::init(init))
        .def("setOptions", &Stepper::setOptions)
        .def("initialize", initialize)
        .def("canonicalize", canonicalize)
        .def("residuals", residuals)
        .def("decompose", decompose)
        .def("solve", solve)
        .def("sensitivities", sensitivities)
        ;
}
