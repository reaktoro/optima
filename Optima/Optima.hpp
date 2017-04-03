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

#pragma once

#include <Optima/Common/Exception.hpp>
#include <Optima/Common/Index.hpp>
#include <Optima/Common/Optional.hpp>
#include <Optima/Common/Outputter.hpp>
#include <Optima/Common/Result.hpp>
#include <Optima/Common/SetUtils.hpp>
#include <Optima/Common/Timing.hpp>
#include <Optima/Core/OptimumOptions.hpp>
#include <Optima/Core/OptimumParams.hpp>
#include <Optima/Core/OptimumProblem.hpp>
#include <Optima/Core/OptimumResult.hpp>
#include <Optima/Core/OptimumSolver.hpp>
#include <Optima/Core/OptimumState.hpp>
#include <Optima/Core/OptimumStepper.hpp>
#include <Optima/Core/OptimumStructure.hpp>
#include <Optima/Core/SaddlePointMatrix.hpp>
#include <Optima/Core/SaddlePointOptions.hpp>
#include <Optima/Core/SaddlePointProblem.hpp>
#include <Optima/Core/SaddlePointResult.hpp>
#include <Optima/Core/SaddlePointSolver.hpp>
#include <Optima/Core/SaddlePointUtils.hpp>
#include <Optima/Math/BlockDiagonalMatrix.hpp>
#include <Optima/Math/Canonicalizer.hpp>
#include <Optima/Math/EigenExtern.hpp>
#include <Optima/Math/Matrix.hpp>
#include <Optima/Math/Utils.hpp>

#ifdef CLING
#include <Optima/Common/Exception.cpp>
#include <Optima/Common/Outputter.cpp>
#include <Optima/Common/Result.cpp>
#include <Optima/Common/Timing.cpp>
#include <Optima/Core/OptimumOptions.cpp>
#include <Optima/Core/OptimumParams.cpp>
#include <Optima/Core/OptimumProblem.cpp>
#include <Optima/Core/OptimumResult.cpp>
#include <Optima/Core/OptimumSolver.cpp>
#include <Optima/Core/OptimumStepper.cpp>
#include <Optima/Core/OptimumStructure.cpp>
#include <Optima/Core/SaddlePointMatrix.cpp>
#include <Optima/Core/SaddlePointProblem.cpp>
#include <Optima/Core/SaddlePointResult.cpp>
#include <Optima/Core/SaddlePointSolver.cpp>
#include <Optima/Core/SaddlePointUtils.cpp>
#include <Optima/Math/BlockDiagonalMatrix.cpp>
#include <Optima/Math/Canonicalizer.cpp>
#include <Optima/Math/EigenExtern.cpp>
#include <Optima/Math/Utils.cpp>
#endif
