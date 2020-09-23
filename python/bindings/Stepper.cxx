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
    auto init = [](Index nx, Index np, Index ny, Index nz, MatrixConstRef4py Ax, MatrixConstRef4py Ap) -> Stepper
    {
        return Stepper({ nx, np, ny, nz, Ax, Ap });
    };

    auto initialize = [](Stepper& self,
        VectorConstRef b,
        VectorConstRef xlower,
        VectorConstRef xupper,
        VectorConstRef plower,
        VectorConstRef pupper,
        VectorRef x,
        Stability& stability)
    {
        self.initialize({ b, xlower, xupper, plower, pupper, x, stability });
    };

    auto canonicalize = [](Stepper& self,
        VectorConstRef x,
        VectorConstRef p,
        VectorConstRef y,
        VectorConstRef z,
        VectorConstRef fx,
        MatrixConstRef4py fxx,
        MatrixConstRef4py fxp,
        MatrixConstRef4py vx,
        MatrixConstRef4py vp,
        MatrixConstRef4py hx,
        MatrixConstRef4py hp,
        VectorConstRef xlower,
        VectorConstRef xupper,
        VectorConstRef plower,
        VectorConstRef pupper,
        Stability& stability)
    {
        self.canonicalize({ x, p, y, z, fx, fxx, fxp, vx, vp, hx, hp, xlower, xupper, plower, pupper, stability });
    };

    auto residuals = [](Stepper& self,
        VectorConstRef x,
        VectorConstRef p,
        VectorConstRef y,
        VectorConstRef z,
        VectorConstRef b,
        VectorConstRef h,
        VectorConstRef v,
        VectorConstRef fx,
        MatrixConstRef4py hx,
        VectorRef rx,
        VectorRef rp,
        VectorRef ry,
        VectorRef rz,
        VectorRef ex,
        VectorRef ep,
        VectorRef ey,
        VectorRef ez,
        VectorRef s)
    {
        self.residuals({ x, p, y, z, b, h, v, fx, hx, rx, rp, ry, rz, ex, ep, ey, ez, s });
    };

    auto decompose = [](Stepper& self)
    {
        self.decompose();
    };

    auto solve = [](Stepper& self,
        VectorConstRef x,
        VectorConstRef p,
        VectorConstRef y,
        VectorConstRef z,
        VectorConstRef fx,
        VectorConstRef b,
        VectorConstRef h,
        VectorConstRef v,
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
        MatrixConstRef4py fxw,
        MatrixConstRef4py hw,
        MatrixConstRef4py bw,
        MatrixConstRef4py vw,
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
