/*
 * Macros.hpp
 *
 *  Created on: 13 Feb 2013
 *      Author: allan
 */

#pragma once

// C++ includes
#include <iostream>

#ifndef NDEBUG

/// An assert macro that allows the output of a helpfull message
#define Assert(condition, message) \
 do { \
     if (! (condition)) { \
         std::cerr << "Assertion `" #condition "` failed in " << __FILE__ \
                   << " line " << __LINE__ << ": " << message << std::endl; \
         std::exit(EXIT_FAILURE); \
       } \
    } while (false)
#else

#define Assert(condition, message) do { } while (false)

#endif // #ifndef NDEBUG


