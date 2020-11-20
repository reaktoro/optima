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
#include <Optima/Result.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/SaddlePointSolver.hpp>
using namespace Optima;

void exportSaddlePointSolver(py::module& m)
{
    auto init = [](Index nx, Index np, Index ny, Index nz, MatrixView4py Ax, MatrixView4py Ap) -> SaddlePointSolver
    {
        return SaddlePointSolver({ nx, np, ny, nz, Ax, Ap });
    };

    auto canonicalize = [](SaddlePointSolver& self, MatrixView4py Hxx, MatrixView4py Hxp, MatrixView4py Hpx, MatrixView4py Hpp, MatrixView4py Jx, MatrixView4py Jp, IndicesView ju, VectorView wx)
    {
        return self.canonicalize({ Hxx, Hxp, Hpx, Hpp, Jx, Jp, ju, wx });
    };

    auto rhs1 = [](SaddlePointSolver& self, VectorView ax, VectorView ap, VectorView ay, VectorView az)
    {
        return self.rhs({ ax, ap, ay, az });
    };

    auto rhs2 = [](SaddlePointSolver& self, VectorView fx, VectorView x, VectorView p, VectorView y, VectorView z, VectorView v, VectorView h, VectorView b)
    {
        return self.rhs({ fx, x, p, y, z, v, h, b });
    };

    auto decompose = [](SaddlePointSolver& self)
    {
        return self.decompose();
    };

    auto solve = [](SaddlePointSolver& self, VectorRef sx, VectorRef sp, VectorRef sy, VectorRef sz)
    {
        self.solve({ sx, sp, sy, sz });
    };

    auto multiply = [](SaddlePointSolver& self, VectorView rx, VectorView rp, VectorView ry, VectorView rz, VectorRef ax, VectorRef ap, VectorRef ay, VectorRef az)
    {
        self.multiply({ rx, rp, ry, rz, ax, ap, ay, az });
    };

    auto transposeMultiply = [](SaddlePointSolver& self, VectorView rx, VectorView rp, VectorView ry, VectorView rz, VectorRef ax, VectorRef ap, VectorRef ay, VectorRef az)
    {
        self.transposeMultiply({ rx, rp, ry, rz, ax, ap, ay, az });
    };

    py::class_<SaddlePointSolverState>(m, "SaddlePointSolverState")
        .def_readonly("dims", &SaddlePointSolverState::dims)
        .def_readonly("js", &SaddlePointSolverState::js)
        .def_readonly("jbs", &SaddlePointSolverState::jbs)
        .def_readonly("jns", &SaddlePointSolverState::jns)
        .def_readonly("ju", &SaddlePointSolverState::ju)
        .def_readonly("jbu", &SaddlePointSolverState::jbu)
        .def_readonly("jnu", &SaddlePointSolverState::jnu)
        .def_readonly("R", &SaddlePointSolverState::R)
        .def_readonly("Hss", &SaddlePointSolverState::Hss)
        .def_readonly("Hsp", &SaddlePointSolverState::Hsp)
        .def_readonly("Vps", &SaddlePointSolverState::Vps)
        .def_readonly("Vpp", &SaddlePointSolverState::Vpp)
        .def_readonly("As", &SaddlePointSolverState::As)
        .def_readonly("Au", &SaddlePointSolverState::Au)
        .def_readonly("Ap", &SaddlePointSolverState::Ap)
        .def_readonly("Js", &SaddlePointSolverState::Js)
        .def_readonly("Jp", &SaddlePointSolverState::Jp)
        .def_readonly("Sbsns", &SaddlePointSolverState::Sbsns)
        .def_readonly("Sbsp", &SaddlePointSolverState::Sbsp)
        .def_readonly("as", &SaddlePointSolverState::as)
        .def_readonly("au", &SaddlePointSolverState::au)
        .def_readonly("ap", &SaddlePointSolverState::ap)
        .def_readonly("ay", &SaddlePointSolverState::ay)
        .def_readonly("az", &SaddlePointSolverState::az)
        ;

    py::class_<SaddlePointSolver>(m, "SaddlePointSolver")
        .def(py::init(init))
        .def("setOptions", &SaddlePointSolver::setOptions)
        .def("options", &SaddlePointSolver::options)
        .def("canonicalize", canonicalize)
        .def("decompose", decompose)
        .def("rhs", rhs1)
        .def("rhs", rhs2)
        .def("solve", solve)
        .def("multiply", multiply)
        .def("transposeMultiply", transposeMultiply)
        .def("state", &SaddlePointSolver::state, py::return_value_policy::reference_internal)
        ;
}
