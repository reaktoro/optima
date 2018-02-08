// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
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

#include <../PyOptima/PyOptima/Common/PyMatrix.hpp"

#include <boost/python.hpp>
namespace py = boost::python;

// Optima includes
#include <Optima/Math/Matrix.hpp>

namespace Optima {

auto export_Matrix() -> void
{
    // Export the typedef Vector = VectorXd
    py::scope().attr("Vector") = py::scope().attr("VectorXd");

    // Export the typedef Matrix = MatrixXd
    py::scope().attr("Matrix") = py::scope().attr("MatrixXd");
}

} // namespace Optima
