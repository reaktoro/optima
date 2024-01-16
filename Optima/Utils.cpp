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

#include "Utils.hpp"

// C++ includes
#include <cmath>
#include <limits>

namespace Optima {

auto largestStep(const Vector& p, const Vector& dp) -> double
{
    Vector res = -p.array() / dp.array();
    double alpha = infinity();
    for(unsigned i = 0; i < res.size(); ++i)
        if(res[i] > 0.0 && res[i] < alpha)
            alpha = res[i];
    return alpha;
}

auto largestStep(const Vector& p, const Vector& dp, const Vector& plower, const Vector& pupper) -> double
{
    using std::isfinite;
    using std::min;
    auto alpha = infinity();
    for(auto i = 0; i < p.size(); ++i)
    {
        assert(p[i] >= plower[i]);
        assert(p[i] <= pupper[i]);
        if(isfinite(plower[i]) && p[i] + dp[i] < plower[i])
        {
            const auto alphai = (plower[i] - p[i])/dp[i]; // from p[i] + alpha[i]*dp[i] = plower[i]
            alpha = min(alpha, alphai);
            continue;
        }
        if(isfinite(pupper[i]) && p[i] + dp[i] > pupper[i])
        {
            const auto alphai = (pupper[i] - p[i])/dp[i]; // from p[i] + alpha[i]*dp[i] = pupper[i]
            alpha = min(alpha, alphai);
            continue;
        }
    }
    return alpha;
}

auto performConservativeStep(Vector& p, const Vector& dp, const Vector& plower, const Vector& pupper) -> double
{
    // Assert all vectors have consistent dimention
    const auto size = p.size();
    assert(dp.size() == size);
    assert(plower.size() == size);
    assert(pupper.size() == size);

    using std::isfinite;
    using std::min;

    // The index of the variable in p that has largest lower/upper bound
    // violation, which will be attached to its bound if applicable.
    auto j = p.size();

    // The factor used to scale the step dp so that variable p[j] is attached
    // to its lower or upper bound (affecting all other variables).
    auto alpha = 1.0;

    // The integer that indicates if p[j] should be attached to its lower bound
    // (-1), to its upper bound (+1), or j is not applicable (0).
    auto lu = 0;

    // Identify index j and alpha factor in the loop below
    for(auto i = 0; i < size; ++i)
    {
        assert(p[i] >= plower[i]);
        assert(p[i] <= pupper[i]);
        if(p[i] == plower[i] && dp[i] < 0.0)
        {
            // dp[i] = 0.0;
            continue;
        }
        if(p[i] == pupper[i] && dp[i] > 0.0)
        {
            // dp[i] = 0.0;
            continue;
        }
        if(isfinite(plower[i]) && p[i] + dp[i] < plower[i])
        {
            const auto alphai = (plower[i] - p[i])/dp[i]; // from p[i] + alpha[i]*dp[i]
            if(alphai < alpha)
            {
                alpha = std::min(alphai, 1.0); // just in case we get alphai = 1 + eps, so that min(alphai, 1) produces 1
                j = i;
                lu = -1;
            }
            continue;
        }
        if(isfinite(pupper[i]) && p[i] + dp[i] > pupper[i])
        {
            const auto alphai = (pupper[i] - p[i])/dp[i]; // from p[i] + alpha[i]*dp[i]
            if(alphai < alpha)
            {
                alpha = std::min(alphai, 1.0); // just in case we get alphai = 1 + eps, so that min(alphai, 1) produces 1
                j = i;
                lu = +1;
            }
            continue;
        }
    }

    // If needed, perform the conservative step using alpha
    if(lu != 0)
    {
        // Compute p' = p + alpha*dp
        p += alpha * dp;

        // Ensure exact bound attachment for p[j]. This is needed because
        // round-off errors in p[j] + alpha*dp[j] can prevent that.
        p[j] = lu == -1 ? plower[j] : pupper[j];
    }
    // Compute p' = p + dp
    else p += dp;

    // Loop once more through all variables and attach any bound violating
    // variable to its closest bound. This is also done in case round-off
    // errors in p + alpha*dp produced slightly bound violations.
    p.noalias() = p.cwiseMax(plower);
    p.noalias() = p.cwiseMin(pupper);

    return alpha;
}

auto performAggressiveStep(Vector& p, const Vector& dp, const Vector& plower, const Vector& pupper) -> void
{
    // Assert all vectors have consistent dimention
    const auto size = p.size();
    assert(dp.size() == size);
    assert(plower.size() == size);
    assert(pupper.size() == size);

    // Perform full step so that p' = p + dp
    p += dp;

    // Ensure no entry in `p'` violates lower/upper bounds
    p.noalias() = p.cwiseMax(plower);
    p.noalias() = p.cwiseMin(pupper);
}

auto stepUpToBounds(const Vector& p, const Vector& dp, const Vector& plower, const Vector& pupper) -> double
{
    using std::isfinite;
    using std::min;
    Vector res = -p.array() / dp.array();
    double alpha = infinity();
    for(auto i = 0; i < res.size(); ++i)
    {
        assert(p[i] >= plower[i]);
        assert(p[i] <= pupper[i]);
        const auto pi = p[i] + dp[i];
        if(isfinite(plower[i]) && p[i] + dp[i] < plower[i])
        {
            const auto alphai = (plower[i] - p[i])/dp[i]; // from p[i] + alpha[i]*dp[i] = plower[i] => alpha[i] = (plower[i] - p[i])/dp[i]
            alpha = min(alpha, alphai);
            continue;
        }
        if(isfinite(pupper[i]) && p[i] + dp[i] > pupper[i])
        {
            const auto alphai = (pupper[i] - p[i])/dp[i]; // from p[i] + alpha[i]*dp[i] = pupper[i] => alpha[i] = (pupper[i] - p[i])/dp[i]
            alpha = min(alpha, alphai);
            continue;
        }

        if(pi > pupper[i])
            alpha = res[i];
    }
    return alpha;
}

auto fractionToTheBoundary(const Vector& p, const Vector& dp, double tau) -> double
{
    Index i;
    return fractionToTheBoundary(p, dp, tau, i);
}

auto fractionToTheBoundary(const Vector& p, const Vector& dp, double tau, Index& ilimiting) -> double
{
    ilimiting = p.size();
    double alpha_max = 1.0;
    for(int i = 0; i < p.size(); ++i)
    {
        if(dp[i] < 0.0)
        {
            const double alpha_trial = -tau*p[i]/dp[i];
            if(alpha_trial < alpha_max)
            {
                alpha_max = alpha_trial;
                ilimiting = i;
            }
        }
    }

    return alpha_max;
}

auto fractionToTheBoundary(const Vector& p, const Vector& dp, const Matrix& C, const Vector& r, double tau) -> double
{
    // The number of linear inequality constraints
    const Index m = C.rows();

    // Check if there is any inequality constraint and return 1.0 if not
    if(m == 0) return 1.0;

    // Otherwise, compute max(alpha)
    double alpha_max = 1.0;
    for(Index i = 0; i < m; ++i)
    {
        const double tmp = C.row(i).dot(dp);
        if(tmp < 0.0)
        {
            const double alpha_trial = -tau*(C.row(i).dot(p) - r[i])/tmp;
            if(alpha_trial < alpha_max)
                alpha_max = alpha_trial;
        }
    }

    return alpha_max;
}

auto fractionToTheLowerBoundary(const Vector& p, const Vector& dp, const Vector& lower, double tau) -> double
{
    double alpha_max = 1.0;
    for(unsigned i = 0; i < p.size(); ++i)
        if(dp[i] < 0.0) alpha_max = std::min(alpha_max, -tau*(p[i] - lower[i])/dp[i]);
    return alpha_max;
}

auto lessThan(double lhs, double rhs, double baseval) -> bool
{
    const double epsilon = std::numeric_limits<double>::epsilon();
    return lhs < rhs + 10.0 * epsilon * std::abs(baseval);
}

auto greaterThan(double lhs, double rhs, double baseval) -> bool
{
    const double epsilon = std::numeric_limits<double>::epsilon();
    return lhs > rhs - 10.0 * epsilon * std::abs(baseval);
}

auto bfgs() -> std::function<Matrix(const Vector&, const Vector&)>
{
    Vector x0;
    Vector g0;
    Vector dx;
    Vector dg;
    Matrix H;

    std::function<Matrix(const Vector&, const Vector&)> f = [=](const Vector& x, const Vector& g) mutable
    {
        if(x0.size() == 0)
        {
            x0.noalias() = x;
            g0.noalias() = g;
            H = diag(x);
            return H;
        }

        dx.noalias() = x - x0;
        dg.noalias() = g - g0;
        x0.noalias() = x;
        g0.noalias() = g;

        const auto n = x.size();
        const auto a = dx.dot(dg);
        const auto I = Eigen::identity(n, n);

        H = (I - dx*tr(dg)/a)*H*(I - dg*tr(dx)/a) + dx*tr(dx)/a;

        return H;
    };

    return f;
}

auto minimizeGoldenSectionSearch(const std::function<double(double)>& f, double tol) -> double
{
    //---------------------------------------------------------------
    // Reference: http://en.wikipedia.org/wiki/Golden_section_search
    //---------------------------------------------------------------

    // The golden ratio
    const double phi = 0.61803398875;

    double a = 0.0;
    double b = 1.0;

    double c = 1 - phi;
    double d = phi;

    if(std::abs(c - d) < tol)
        return (b + a)/2.0;

    double fc = f(c);
    double fd = f(d);

    while(std::abs(c - d) > tol)
    {
        if(fc < fd)
        {
            b = d;
            d = c;
            c = b - phi*(b - a);
            fd = fc;
            fc = f(c);
        }
        else
        {
            a = c;
            c = d;
            d = a + phi*(b - a);
            fc = fd;
            fd = f(d);
        }
    }

    return (b + a)/2.0;
}


auto minimizeGoldenSectionSearch(const std::function<double(double)>& f, double a, double b, double tol) -> double
{
    auto g = [=](double x)
    {
        return f(a + x*(b - a));
    };

    const double xmin = minimizeGoldenSectionSearch(g, tol);
    return a + xmin*(b - a);
}

auto minimizeBrent(const std::function<double(double)>& f, double min, double max, double tolerance, unsigned maxiters) -> double
{
    //-------------------------------------------------------------------
    // The code below was adapted from boost library, found at header
    // boost/math/tools/minima.hpp under the name brent_find_minima.
    //-------------------------------------------------------------------
    double x; // minima so far
    double w; // second best point
    double v; // previous value of w
    double u; // most recent evaluation point
    double delta; // The distance moved in the last step
    double delta2; // The distance moved in the step before last
    double fu, fv, fw, fx; // function evaluations at u, v, w, x
    double mid; // midpoint of min and max
    double fract1, fract2; // minimal relative movement in x

    const double golden = 0.3819660;// golden ratio, don't need too much precision here!

    x = w = v = max;
    fw = fv = fx = f(x);
    delta2 = delta = 0;

    unsigned count = maxiters;

    do
    {
        // get midpoint
        mid = (min + max)/2;

        // work out if we're done already:
        fract1 = tolerance * fabs(x) + tolerance/4;
        fract2 = 2 * fract1;

        if(fabs(x - mid) <= (fract2 - (max - min)/2))
            break;

        if(fabs(delta2) > fract1)
        {
            // try and construct a parabolic fit:
            double r = (x - w) * (fx - fv);
            double q = (x - v) * (fx - fw);
            double p = (x - v) * q - (x - w) * r;
            q = 2 * (q - r);
            if(q > 0)
            p = -p;
            q = fabs(q);
            double td = delta2;
            delta2 = delta;

            // determine whether a parabolic step is acceptible or not:
            if((fabs(p) >= fabs(q * td / 2)) || (p <= q * (min - x)) || (p >= q * (max - x)))
            {
                // nope, try golden section instead
                delta2 = (x >= mid) ? min - x : max - x;
                delta = golden * delta2;
            }
            else
            {
                // when, parabolic fit:
                delta = p / q;
                u = x + delta;
                if(((u - min) < fract2) || ((max- u) < fract2))
                delta = (mid - x) < 0 ? -fabs(fract1) : fabs(fract1);
            }
        }
        else
        {
            // golden section:
            delta2 = (x >= mid) ? min - x : max - x;
            delta = golden * delta2;
        }
        // update current position:
        u = (fabs(delta) >= fract1) ? x + delta : (delta > 0 ? x + fabs(fract1) : x - fabs(fract1));
        fu = f(u);
        if(fu <= fx)
        {
            // good new point is an improvement!
            // update brackets:
            if(u >= x)
            min = x;
            else
            max = x;
            // update control points:
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        }
        else
        {
            // Oh dear, point u is worse than what we have already,
            // even so it *must* be better than one of our endpoints:
            if(u < x)
            min = u;
            else
            max = u;
            if((fu <= fw) || (w == x))
            {
                // however it is at least second best:
                v = w;
                w = u;
                fv = fw;
                fw = fu;
            }
            else if((fu <= fv) || (v == x) || (v == w))
            {
                // third best:
                v = u;
                fv = fu;
            }
        }

    } while(--count);

    maxiters -= count;

    return x;
}

auto inverseShermanMorrison(const Matrix& invA, const Vector& D) -> Matrix
{
    Matrix invM = invA;
    for(auto i = 0; i < D.rows(); ++i)
        invM = invM - (D[i]/(1 + D[i]*invM(i, i)))*invM.col(i)*invM.row(i);
    return invM;
}

/// Return the numerator and denominator of the rational number closest to `x`.
/// This methods expects `0 <= x <= 1`.
/// @param x The number for which the closest rational number is sought.
/// @param n The maximum denominator that the rational number can have.
auto farey(double x, std::size_t n) -> std::tuple<long, long>
{
    long a = 0, b = 1;
    long c = 1, d = 1;
    while(b <= n && d <= n)
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

auto rational(double x, unsigned n) -> std::tuple<long, long>
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

auto rationalize(double* data, unsigned size, unsigned maxden) -> void
{
    auto ratio = [&](double val) -> double
    {
        auto pair = Optima::rational(val, maxden);
        return static_cast<double>(std::get<0>(pair))/std::get<1>(pair);
    };
    std::transform(data, data + size, data, ratio);
}

/// Replace residual round-off errors by zeros in a vector.
auto cleanResidualRoundoffErrors(VectorRef vec) -> void
{
    if(vec.size() == 0) return;
    const auto eps = std::numeric_limits<double>::epsilon();
    const auto max = vec.array().abs().maxCoeff();
    const auto rows = vec.rows();
    const auto threshold = eps * max * rows;
    for(auto i = 0; i  < rows; ++i)
        if(std::abs(vec[i]) <= threshold)
            vec[i] = 0.0;
}

/// Replace residual round-off errors by zeros in a matrix.
auto cleanResidualRoundoffErrors(MatrixRef mat) -> void
{
    if(mat.size() == 0) return;
    const auto eps = std::numeric_limits<double>::epsilon();
    const auto max = mat.array().abs().maxCoeff();
    const auto rows = mat.rows();
    const auto cols = mat.cols();
    const auto threshold = eps * max * std::max(rows, cols);
    for(auto i = 0; i  < rows; ++i)
        for(auto j = 0; j  < cols; ++j)
            if(std::abs(mat(i, j)) <= threshold)
                mat(i, j) = 0.0;
}

auto multiplyMatrixVectorWithoutResidualRoundOffError(MatrixView A, VectorView x) -> Vector
{
    // In this method, we use b' = |A|*|x| as a reference to determine which
    // small entries in b should be regarded as residual round-off error. The
    // idea is that if b[i] is truly small, and not a result of round-off
    // errors, then b'[i] should also be small.

    assert(A.cols() == x.rows());

    Vector b = A * x;
    const auto eps = std::numeric_limits<double>::epsilon();
    for(auto i = 0; i < b.size(); ++i)
    {
        const double ref = A.row(i).cwiseAbs() * x.cwiseAbs();
        if(std::abs(b[i]) < ref * eps)
            b[i] = 0.0;
    }
    return b;
}

auto matrixStructure(MatrixView mat) -> MatrixStructure
{
    if(isDenseMatrix(mat)) return MatrixStructure::Dense;
    if(isDiagonalMatrix(mat)) return MatrixStructure::Diagonal;
    return MatrixStructure::Zero;
}

auto isZeroMatrix(MatrixView mat) -> bool
{
    return mat.size() == 0;
}

auto isDiagonalMatrix(MatrixView mat) -> bool
{
    return mat.size() > 1 && mat.cols() == 1;
}

auto isDenseMatrix(MatrixView mat) -> bool
{
    return mat.size() > 0 && mat.rows() == mat.cols();
}

auto operator<<=(MatrixRef mat, MatrixView other) -> MatrixRef
{
    switch(matrixStructure(other)) {
    case MatrixStructure::Dense: mat = other; break;
    case MatrixStructure::Diagonal: mat = diag(other.col(0)); break;
    case MatrixStructure::Zero: break;
    }
    return mat;
}

auto ensureMinimumDimension(Matrix& mat, Index rows, Index cols) -> void
{
    const auto m = std::max(mat.rows(), rows);
    const auto n = std::max(mat.cols(), cols);
    mat.resize(m, n);
}

} // namespace Optima
