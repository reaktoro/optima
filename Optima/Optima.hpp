// Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
//
// Copyright (C) 2014-2016 Allan Leal
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
#include <Optima/Common/Outputter.hpp>
#include <Optima/Common/SetUtils.hpp>
#include <Optima/Common/Timing.hpp>
#include <Optima/Core/Filter.hpp>
#include <Optima/Core/Hessian.hpp>
#include <Optima/Core/Jacobian.hpp>
#include <Optima/Core/KktSolver.hpp>
#include <Optima/Core/NonlinearSolver.hpp>
#include <Optima/Core/OptimumMethod.hpp>
#include <Optima/Core/OptimumOptions.hpp>
#include <Optima/Core/OptimumProblem.hpp>
#include <Optima/Core/OptimumResult.hpp>
#include <Optima/Core/OptimumSolver.hpp>
#include <Optima/Core/OptimumSolverBase.hpp>
#include <Optima/Core/OptimumSolverIpNewton.hpp>
#include <Optima/Core/OptimumState.hpp>
#include <Optima/Core/Regularizer.hpp>
#include <Optima/Core/Utils.hpp>
#include <Optima/Math/LU.hpp>
#include <Optima/Math/Matrix.hpp>
#include <Optima/Math/Utils.hpp>
