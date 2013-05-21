/*
 * Demo3.cpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#include <iostream>
#include <Optima.hpp>

double f(double x) { return -3*x*x*x + 9*x + 8; }

int main()
{
    Optima::BrentSolver brent;
    brent.SetFunction(f);
    brent.Solve(-3, 0, 3);

    Optima::GoldenSolver golden;
    golden.SetFunction(f);
    golden.Solve(-3, 0, 3);

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
