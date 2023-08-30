# Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
#
# Copyright (C) 2020 Allan Leal
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.

from optima import *

import numpy as npy

from numpy import (
    random,
    linalg,
)

from numpy.testing import (
    assert_allclose,
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
)

import pytest

from pytest import approx

import math

# Ensure proper options for printing numpy arrays
npy.set_printoptions(linewidth=1000)

npy.random.seed(0)
