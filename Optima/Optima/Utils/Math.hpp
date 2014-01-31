/*
 * Math.hpp
 *
 *  Created on: 5 Apr 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <cmath>
#include <limits>

// Eigen includes
#include <Eigen/Core>
using namespace Eigen;

namespace Optima {

/// The double representation of infinity
const double INF = std::numeric_limits<double>::infinity();

template<typename T>
inline bool isnan(const T& x)
{
    return std::isnan(x);
}

template<typename T>
inline bool isinf(const T& x)
{
    return std::isinf(x);
}

template<typename T>
inline bool isfinite(const T& x)
{
    return std::isfinite(x);
}

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
inline bool isnan(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& x)
{
    return x.unaryExpr(static_cast<bool(*)(const Scalar&)>(isnan)).any();
}

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
inline bool isinf(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& x)
{
    return x.unaryExpr(static_cast<bool(*)(const Scalar&)>(isinf)).any();
}

template <typename Scalar, int Rows, int Cols, int Options, int MaxRows, int MaxCols>
inline bool isfinite(const Matrix<Scalar, Rows, Cols, Options, MaxRows, MaxCols>& x)
{
    return x.unaryExpr(static_cast<bool(*)(const Scalar&)>(isfinite)).all();
}

template <typename T>
inline double positive(const T& x)
{
    return (x > 0.0) ? x : std::numeric_limits<T>::infinity();
}

template <typename T>
inline double negative(const T& x)
{
    return (x < 0.0) ? x : -std::numeric_limits<T>::infinity();
}

}  /* namespace Optima */