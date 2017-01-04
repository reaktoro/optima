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

#include "Utils.hpp"

namespace Optima {

auto inverseShermanMorrison(const MatrixXd& invA, const VectorXd& D) -> MatrixXd
{
    MatrixXd invM = invA;
    for(unsigned i = 0; i < D.rows(); ++i)
        invM = invM - (D[i]/(1 + D[i]*invM(i, i)))*invM.col(i)*invM.row(i);
    return invM;
}

/// Return the numerator and denominator of the rational number closest to `x`.
/// This methods expects `0 <= x <= 1`.
/// @param x The number for which the closest rational number is sought.
/// @param n The maximum denominator that the rational number can have.
auto farey(double x, unsigned n) -> std::tuple<long, long>
{
    long a = 0, b = 1;
    long c = 1, d = 1;
    while(b <= n and d <= n)
    {
        double mediant = double(a+c)/(b+d);
        if(x == mediant) {
            if(b + d <= n) return std::make_tuple(a+c, b+d);
            if(d > b) return std::make_tuple(c, d);
            return std::make_tuple(a, b);
        }
        if(x > mediant) {
            a = a+c;
            b = b+d;
        }
        else {
            c = a+c;
            d = b+d;
        }
    }

    return (b > n) ? std::make_tuple(c, d) : std::make_tuple(a, b);
}

auto rationalize(double x, unsigned n) -> std::tuple<long, long>
{
    long a, b, sign = (x >= 0) ? +1 : -1;
    if(std::abs(x) > 1.0) {
        std::tie(a, b) = farey(1.0/std::abs(x), n);
        return std::make_tuple(sign*b, a);
    }
    else {
        std::tie(a, b) = farey(std::abs(x), n);
        return std::make_tuple(sign*a, b);
    }
}

} // namespace Optima
