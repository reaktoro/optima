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

#include <Optima/Optima.hpp>
using namespace Optima;

int main(int argc, char **argv)
{
    Matrix A = Matrix::Random(4, 10);

    std::cout << std::fixed;
    std::cout << "A = \n" << A << std::endl;

    Echelonizer echelonizer(A);

    std::cout << "C = \n" << echelonizer.C() << std::endl;
    std::cout << "basic variables = " << echelonizer.indicesBasicVariables().transpose() << std::endl;
    std::cout << "non-basic variables = " << echelonizer.indicesNonBasicVariables().transpose() << std::endl;

    // Swap the first non-basic variable with the first basic variable
    echelonizer.updateWithSwapBasicVariable(0, 0);

    std::cout << "updated C = \n" << echelonizer.C() << std::endl;
    std::cout << "updated basic variables = " << echelonizer.indicesBasicVariables().transpose() << std::endl;
    std::cout << "updated non-basic variables = " << echelonizer.indicesNonBasicVariables().transpose() << std::endl;
}
