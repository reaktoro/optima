// // Optima is a C++ library for solving linear and non-linear constrained optimization problems
// //
// // Copyright (C) 2014-2018 Allan Leal
// //
// // This program is free software: you can redistribute it and/or modify
// // it under the terms of the GNU General Public License as published by
// // the Free Software Foundation, either version 3 of the License, or
// // (at your option) any later version.
// //
// // This program is distributed in the hope that it will be useful,
// // but WITHOUT ANY WARRANTY; without even the implied warranty of
// // MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// // GNU General Public License for more details.
// //
// // You should have received a copy of the GNU General Public License
// // along with this program. If not, see <http://www.gnu.org/licenses/>.

// // pybind11 includes
// #include <pybind11/pybind11.h>
// #include <pybind11/eigen.h>
// #include <pybind11/functional.h>
// namespace py = pybind11;

// // Optima includes
// #include <Optima/BasicSolver.hpp>
// #include <Optima/Options.hpp>
// #include <Optima/Result.hpp>
// #include <Optima/Stability.hpp>
// using namespace Optima;

// void exportBasicSolver(py::module& m)
// {
//     auto init = [](Index nx, Index np, Index ny, Index nz, MatrixView4py Ax, MatrixView4py Ap) -> BasicSolver
//     {
//         return BasicSolver({ nx, np, ny, nz, Ax, Ap });
//     };

//     auto solve = [](BasicSolver& self,
//         ObjectiveFunction4py const& obj4py,
//         ConstraintFunction4py const& h4py,
//         ConstraintFunction4py const& v4py,
//         VectorView b,
//         VectorView xlower,
//         VectorView xupper,
//         VectorView plower,
//         VectorView pupper,
//         VectorRef x,
//         VectorRef p,
//         VectorRef y,
//         VectorRef z,
//         VectorRef s,
//         Stability& stability) -> Result
//     {
//         auto obj = convert(obj4py);
//         auto h = convert(h4py);
//         auto v = convert(v4py);
//         return self.solve({ obj, h, v, b, xlower, xupper, plower, pupper, x, p, y, z, s, stability });
//     };

//     Matrix tmp_xw, tmp_pw, tmp_yw, tmp_zw, tmp_sw;
//     auto sensitivities = [=](BasicSolver& self,
//         MatrixView4py fxw,
//         MatrixView4py hw,
//         MatrixView4py bw,
//         MatrixView4py vw,
//         Stability const& stability,
//         MatrixRef4py xw,
//         MatrixRef4py pw,
//         MatrixRef4py yw,
//         MatrixRef4py zw,
//         MatrixRef4py sw) mutable
//     {
//         tmp_xw.resize(xw.rows(), xw.cols());
//         tmp_pw.resize(pw.rows(), pw.cols());
//         tmp_yw.resize(yw.rows(), yw.cols());
//         tmp_zw.resize(zw.rows(), zw.cols());
//         tmp_sw.resize(sw.rows(), sw.cols());
//         self.sensitivities({ fxw, hw, bw, vw, stability, tmp_xw, tmp_pw, tmp_yw, tmp_zw, tmp_sw });
//         xw = tmp_xw;
//         pw = tmp_pw;
//         yw = tmp_yw;
//         zw = tmp_zw;
//         sw = tmp_sw;
//     };

//     py::class_<BasicSolver>(m, "BasicSolver")
//         .def(py::init(init))
//         .def("setOptions", &BasicSolver::setOptions)
//         .def("solve", solve)
//         .def("sensitivities", sensitivities)
//         ;
// }
