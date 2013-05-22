/*
 * Demo3.cpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#include <cmath>
#include <iostream>
#include <Optima.hpp>

double f(double x) { return 3*x*x + 6*x + 8; }
double g(double x) { return x*std::sin(x); }

int main()
{
    double a = -500;
    double b = +400;
    double c = 0;
    unsigned niters;

    Optima::Bracket(f, a, b, c, &niters);

    std::cout << "a, b, c: " << std::endl;
    std::cout << a << std::endl;
    std::cout << b << std::endl;
    std::cout << c << std::endl;
    std::cout << "f(a), f(b), f(c): " << std::endl;
    std::cout << f(a) << std::endl;
    std::cout << f(b) << std::endl;
    std::cout << f(c) << std::endl;
    std::cout << "niters:\n" << niters << std::endl;
    std::cout << std::endl;

    Optima::BrentSolver brent;
    brent.SetFunction(f);
    brent.Solve(a, b, c);

    Optima::GoldenSolver golden;
    golden.SetFunction(f);
    golden.Solve(a, b, c);

    const auto& brent_res = brent.GetResult();
    const auto& brent_sta = brent.GetStatistics();

    const auto& golden_res = golden.GetResult();
    const auto& golden_sta = golden.GetStatistics();

    std::cout << "niters1: " << golden_sta.niters << std::endl;
    std::cout << "xmin1: " << golden_res.xmin << std::endl;
    std::cout << "fmin1: " << golden_res.fmin << std::endl;
    std::cout << std::endl;
    std::cout << "niters2: " << brent_sta.niters << std::endl;
    std::cout << "xmin2: " << brent_res.xmin << std::endl;
    std::cout << "fmin2: " << brent_res.fmin << std::endl;
}
