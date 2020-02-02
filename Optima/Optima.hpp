// Optima is a C++ library for solving linear and non-linear constrained optimization problems
//
// Copyright (C) 2014-2018 Allan Leal
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

// Optima includes
#include <Optima/BasicSolver.hpp>
#include <Optima/Canonicalizer.hpp>
#include <Optima/ConstraintFunction.hpp>
#include <Optima/Exception.hpp>
#include <Optima/ExtendedCanonicalizer.hpp>
#include <Optima/Index.hpp>
#include <Optima/IndexUtils.hpp>
#include <Optima/Matrix.hpp>
#include <Optima/Number.hpp>
#include <Optima/ObjectiveFunction.hpp>
#include <Optima/Options.hpp>
#include <Optima/Outputter.hpp>
#include <Optima/Partition.hpp>
#include <Optima/Problem.hpp>
#include <Optima/Result.hpp>
#include <Optima/SaddlePointOptions.hpp>
#include <Optima/SaddlePointSolver.hpp>
#include <Optima/Solver.hpp>
#include <Optima/State.hpp>
#include <Optima/Stepper.hpp>
#include <Optima/Timing.hpp>
#include <Optima/Utils.hpp>
