// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2017 Allan Leal
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
#include <eigen3/Eigenx/Core>
//#include <eigen3/Eigen/Dense>
using namespace Eigen;
namespace py = pybind11;

//// Optima includes
//#include <PyOptima/Math/PyCanonicalizer.hpp>

namespace Optima {

VectorXd doubleVec(VectorXdConstRef a)
{
    return 2*a;
}

MatrixXd squareMat(MatrixXdConstRef a)
{
    return a.cwiseProduct(a);
}

} // namespace Optima

void exportCanonicalizer(py::module& m);

PYBIND11_MODULE(optima, m)
{
    exportCanonicalizer(m);

//    m.def("doubleVec", [](const Eigen::VectorXd& m) -> Eigen::VectorXd { return 2 * m; });
//    m.def("squareMat", [](Eigen::MatrixXd m) -> Eigen::MatrixXd { return m; });

    m.def("doubleVec", &Optima::doubleVec);
    m.def("squareMat", &Optima::squareMat);
}
