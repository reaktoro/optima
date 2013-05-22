/*
 * Utils.cpp
 *
 *  Created on: 21 May 2013
 *      Author: allan
 */

#include "Utils.hpp"

// C++ includes
#include <algorithm>

// Optima includes
#include <Utils/Math.hpp>

namespace Optima {

void Bracket(const std::function<double(double)>& f, double& a, double& b, double& c, unsigned* niters)
{
    const double GOLD = 1.618034;
    const double GLIMIT = 100.0;
    const double TINY = 1.0e-20;

    double ulim, u, r, q, dum, fa, fb, fc, fu;

    fa = f(a);
    fb = f(b);

    while(not isfinite(fa) or not isfinite(fb))
    {
        if(not isfinite(fa))
            a += (GOLD - 1.0)*(b - a);

        if(not isfinite(fb))
            b -= (GOLD - 1.0)*(b - a);

        fa = f(a);
        fb = f(b);
    }

    fa = f(a);
    fb = f(b);

    if(fb > fa)
    {
        std::swap(a, b);
        std::swap(fa, fb);
    }
    c = b + GOLD * (b - a);
    fc = f(c);

    unsigned iter = 1;

    while(fb > fc)
    {
        r = (b - a) * (fb - fc);
        q = (b - c) * (fb - fa);
        u = b - ((b - c)*q - (b - a)*r)/(2.0*Sign(std::max(std::abs(q - r), TINY), q - r));
        ulim = b + GLIMIT*(c - b);
        if((b - u)*(u - c) > 0.0)
        {
            fu = f(u);
            if(fu < fc)
            {
                a = b;
                b = u;
                fa = fb;
                fb = fu;
                break;
            }
            if(fu > fb)
            {
                c = u;
                fc = fu;
                break;
            }
            u = c + GOLD*(c - b);
            fu = f(u);
        }
        else if((c - u)*(u - ulim) > 0.0)
        {
            fu = f(u);
            if(fu < fc)
            {
                Shift(b, c, u, c + GOLD*(c - b));
                Shift(fb, fc, fu, f(u));
            }
        }
        else if((u - ulim)*(ulim - c) >= 0.0)
        {
            u = ulim;
            fu = f(u);
        }
        else
        {
            u = c + GOLD*(c - b);
            fu = f(u);
        }

        Shift(a, b, c, u);
        Shift(fa, fb, fc, fu);

        ++iter;
    }

    if(niters != 0) *niters = iter;
}

} /* namespace Optima */
