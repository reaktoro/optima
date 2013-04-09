/*
 * Main.cpp
 *
 *  Created on: 22 Mar 2013
 *      Author: allan
 */

// C++ includes
#include <algorithm>
#include <cmath>
#include <list>
#include <tuple>

const double gamma_theta_f = 1.0e-05;
const double gamma_theta_o = 1.0e-05;
const double gamma_theta_c = 1.0e-05;

const double delta = 0.1;

const double xi = 0.05;

const double s_f = 1.1;
const double s_o = 2.3;

const double eta = 1.0e-04;

const double eta_theta_f = 1.0e-04;
const double eta_theta_c = 1.0e-04;

const double delta_mu = 0.1;

const double epsilon = 0.1;

class LineSearchFilter
{
public:
    LineSearchFilter();

    LineSearchFilter(double theta_f, double theta_c, double theta_o);

    bool Contains(double theta_f, double theta_c, double theta_o) const;

    void Add(double theta_f, double theta_c, double theta_o);

private:

    double theta_fmax;

    double theta_cmax;

    double theta_omax;

    std::list<std::tuple<double, double, double>> theta;
};

LineSearchFilter::LineSearchFilter()
{}

LineSearchFilter::LineSearchFilter(double theta_f, double theta_c, double theta_o)
{
    // Initialise the filter bounds
    theta_fmax = 1.0e+04 * std::max(1.0, theta_f);
    theta_cmax = 1.0e+04 * std::max(1.0, theta_c);
    theta_omax = 1.0e+04 * std::max(1.0, theta_o);

    // Initialise the filter
    theta.emplace_back(theta_fmax, theta_cmax, theta_omax);
}

bool LineSearchFilter::Contains(double theta_f, double theta_c, double theta_o) const
{
    // Check if the given vertice is outside the bounds of the current state of the filter
    if(theta_f >= theta_fmax or theta_c >= theta_cmax or theta_o >= theta_omax)
        return true;

    // Check if the given vertice is outside the domain of each filter vertice
    for(const auto& triplet : theta)
        if(theta_f >= std::get<0>(triplet) and
           theta_c >= std::get<1>(triplet) and
           theta_o >= std::get<2>(triplet))
            return true;

    return false;
}

void LineSearchFilter::Add(double theta_f, double theta_c, double theta_o)
{
    // Define the superfluous condition function to remove superfluous vertices from the filter
    auto superfluous = [=](const std::tuple<double, double, double>& entry)
    {
        return theta_f < std::get<0>(entry) and
               theta_c < std::get<1>(entry) and
               theta_o < std::get<2>(entry);
    };

    // Remove all superfluous vertices in the filter
    std::remove_if(theta.begin(), theta.end(), superfluous);

    // Add the new vertices to the filter
    theta.emplace_back(theta_f, theta_c, theta_o);

    // Update the maxmium of the three measures
    theta_fmax = std::get<0>(theta.front());
    theta_cmax = std::get<1>(theta.front());
    theta_omax = std::get<2>(theta.front());

    for(const auto& entry : theta)
    {
        theta_fmax = std::max(theta_fmax, std::get<0>(entry));
        theta_cmax = std::max(theta_cmax, std::get<1>(entry));
        theta_omax = std::max(theta_omax, std::get<2>(entry));
    }
}
