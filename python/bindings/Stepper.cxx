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
    auto init = [](Index nx, Index np, Index ny, Index nz, MatrixView4py Ax, MatrixView4py Ap) -> Stepper
    {
        return Stepper({ nx, np, ny, nz, Ax, Ap });
    };

    auto initialize = [](Stepper& self,
        VectorView b,
        VectorView xlower,
        VectorView xupper,
        VectorView plower,
        VectorView pupper,
        VectorRef x,
        Stability& stability)
    {
        self.initialize({ b, xlower, xupper, plower, pupper, x, stability });
    };

    auto canonicalize = [](Stepper& self,
        VectorView x,
        VectorView p,
        VectorView y,
        VectorView z,
        VectorView fx,
        MatrixView4py fxx,
        MatrixView4py fxp,
        MatrixView4py vx,
        MatrixView4py vp,
        MatrixView4py hx,
        MatrixView4py hp,
        VectorView xlower,
        VectorView xupper,
        VectorView plower,
        VectorView pupper,
        Stability& stability)
    {
        self.canonicalize({ x, p, y, z, fx, fxx, fxp, vx, vp, hx, hp, xlower, xupper, plower, pupper, stability });
    };

    auto residuals = [](Stepper& self,
        VectorView x,
        VectorView p,
        VectorView y,
        VectorView z,
        VectorView b,
        VectorView h,
        VectorView v,
        VectorView fx,
        MatrixView4py hx,
        VectorRef rx,
        VectorRef rp,
        VectorRef rw,
        VectorRef ex,
        VectorRef ep,
        VectorRef ew,
        VectorRef s)
    {
        self.residuals({ x, p, y, z, b, h, v, fx, hx, rx, rp, rw, ex, ep, ew, s });
    };

    auto decompose = [](Stepper& self)
    {
        self.decompose();
    };

    auto solve = [](Stepper& self,
        VectorView x,
        VectorView p,
        VectorView y,
        VectorView z,
        VectorView fx,
        VectorView b,
        VectorView h,
        VectorView v,
        Stability const& stability,
        VectorRef dx,
        VectorRef dp,
        VectorRef dy,
        VectorRef dz)
    {
        self.solve({ x, p, y, z, fx, b, h, v, stability, dx, dp, dy, dz });
    };

    Matrix tmp_xw, tmp_pw, tmp_yw, tmp_zw, tmp_sw;
    auto sensitivities = [=](Stepper& self,
        MatrixView4py fxw,
        MatrixView4py hw,
        MatrixView4py bw,
        MatrixView4py vw,
        Stability const& stability,
        MatrixRef4py xw,
        MatrixRef4py pw,
        MatrixRef4py yw,
        MatrixRef4py zw,
        MatrixRef4py sw) mutable
    {
        tmp_xw.resize(xw.rows(), xw.cols());
        tmp_pw.resize(pw.rows(), pw.cols());
        tmp_yw.resize(yw.rows(), yw.cols());
        tmp_zw.resize(zw.rows(), zw.cols());
        tmp_sw.resize(sw.rows(), sw.cols());
        self.sensitivities({ fxw, hw, bw, vw, stability, tmp_xw, tmp_pw, tmp_yw, tmp_zw, tmp_sw });
        xw = tmp_xw;
        pw = tmp_pw;
        yw = tmp_yw;
        zw = tmp_zw;
        sw = tmp_sw;
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
