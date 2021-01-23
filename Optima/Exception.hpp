// Optima is a unified framework for modeling chemically reactive systems.
//
// Copyright (C) 2014-2020 Allan Leal
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <http://www.gnu.org/licenses/>.

#pragma once

// C++ includes
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace Optima {
namespace internal {

template <typename Arg>
auto _stringfy(std::stringstream& ss, const std::string& sep, const Arg& item) -> void
{
    ss << item;
}

template <typename Arg, typename... Args>
auto _stringfy(std::stringstream& ss, const std::string& sep, const Arg& item, Args... items) -> void
{
    ss << item << sep;
    _stringfy(ss, sep, items...);
}

/// Concatenate the arguments into a string using a given separator string.
template <typename... Args>
auto stringfy(const std::string& sep, Args... items) -> std::string
{
    std::stringstream ss;
    _stringfy(ss, sep, items...);
    return ss.str();
}

template <typename... Args>
auto str(Args... items) -> std::string
{
    return stringfy("", items...);
}

} // namespace internal

/// Issue a warning message if condition is true.
template<typename... Args>
auto warning(bool condition, Args... items) -> void
{
    if(condition)
        std::cerr << "\033[1;33m***WARNING***\n" << internal::str(items...) << "\n\033[0m";
}

/// Raise a runtime error if condition is true.
template<typename... Args>
auto error(bool condition, Args... items) -> void
{
    if(condition)
        throw std::runtime_error(internal::str("\033[1;31m***ERROR***\n", internal::str(items...), "\n\033[0m"));
}

/// Define a macro to print a warning messageif condition is true.
/// @ingroup Common
#define warningif(condition, args...) \
    { \
        if((condition)) { \
            std::cerr << "\033[1;33m***WARNING***\n" << internal::str(args) << "\n\033[0m"; \
        } \
    }

/// Define a macro to raise a runtime exception if condition is true.
/// @ingroup Common
#define errorif(condition, args...) \
    { \
        if((condition)) { \
            throw std::runtime_error(internal::str("\033[1;31m***ERROR***\n", internal::str(args), "\n\033[0m")); \
        } \
    }

} // namespace Optima
