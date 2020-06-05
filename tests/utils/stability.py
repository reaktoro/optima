# Optima is a C++ library for numerical solution of linear and nonlinear programing problems.
#
# Copyright (C) 2014-2018 Allan Leal
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
from numpy import *


def create_expected_stability(A, x, b, z, xlower, xupper):
    """Return an expected Stability object for given state variables.

    Arguments:
        A {array} -- The coefficient matrix A
        x {array} -- The primal variables x
        b {array} -- The right-hand side vector b
        z {array} -- The unstabilities of the variables z = g + tr(W)*y
        xlower {array} -- The lower bounds for x
        xupper {array} -- The upper bounds for x
    """

    n = len(x)
    m = len(b)

    ipositiverows = []
    inegativerows = []
    for i in range(m):
        if min(A[i, :]) >= 0.0:
            ipositiverows.append(i)
        if max(A[i, :]) <= 0.0:
            inegativerows.append(i)

    istrictly_lower_unstable = set()
    istrictly_upper_unstable = set()

    for i in ipositiverows:
        if dot(A[i, :], xlower) >= b[i]:
            for j in range(n):
                if A[i, j] != 0.0:
                    istrictly_lower_unstable.add(j)

    for i in ipositiverows:
        if dot(A[i, :], xupper) <= b[i]:
            for j in range(n):
                if A[i, j] != 0.0:
                    istrictly_upper_unstable.add(j)

    for i in inegativerows:
        if dot(A[i, :], xlower) <= b[i]:
            for j in range(n):
                if A[i, j] != 0.0:
                    istrictly_lower_unstable.add(j)

    for i in inegativerows:
        if dot(A[i, :], xupper) >= b[i]:
            for j in range(n):
                if A[i, j] != 0.0:
                    istrictly_upper_unstable.add(j)

    # Remove possible duplicates in both istrictly_lower_unstable and istrictly_upper_unstable
    istrictly_lower_unstable = istrictly_lower_unstable - istrictly_upper_unstable

    # Convert both istrictly_lower_unstable and istrictly_upper_unstable to lists
    istrictly_lower_unstable = list(istrictly_lower_unstable)
    istrictly_upper_unstable = list(istrictly_upper_unstable)

    ilower_unstable = []
    iupper_unstable = []

    for i in range(n):
        if i in istrictly_lower_unstable or i in istrictly_upper_unstable:
            continue

        if x[i] == xlower[i] and z[i] > 0.0:
            ilower_unstable.append(i)

        if x[i] == xupper[i] and z[i] < 0.0:
            iupper_unstable.append(i)

    iunstable = ilower_unstable + iupper_unstable + istrictly_lower_unstable + istrictly_upper_unstable
    istable = list(set(range(n)) - set(iunstable))

    stability = Stability()
    stability.iordering = istable + iunstable
    stability.ns   = len(istable)
    stability.nlu  = len(ilower_unstable)
    stability.nuu  = len(iupper_unstable)
    stability.nslu = len(istrictly_lower_unstable)
    stability.nsuu = len(istrictly_upper_unstable)

    return stability


def check_stability(actual, expected):
    """Check if actual and expected stability states are equivalent.

    Arguments:
        actual {Stability} -- The Stability object obtained in the calculation
        expected {Stability} -- The Stability object with expected values
    """

    def istable(stability):
        return set(stability.iordering[:stability.ns])

    def ilower_unstable(stability):
        return set(stability.iordering[stability.ns:][:stability.nlu])

    def iupper_unstable(stability):
        return set(stability.iordering[stability.ns:][stability.nlu:][:stability.nuu])

    def istrictly_lower_unstable(stability):
        return set(stability.iordering[stability.ns:][stability.nlu:][stability.nuu:][:stability.nslu])

    def istrictly_upper_unstable(stability):
        return set(stability.iordering[stability.ns:][stability.nlu:][stability.nuu:][stability.nslu:][stability.nsuu:])

    assert istable(actual)                  == istable(expected)
    assert ilower_unstable(actual)          == ilower_unstable(expected)
    assert iupper_unstable(actual)          == iupper_unstable(expected)
    assert istrictly_lower_unstable(actual) == istrictly_lower_unstable(expected)
    assert istrictly_upper_unstable(actual) == istrictly_upper_unstable(expected)

    assert actual.ns   == expected.ns
    assert actual.nlu  == expected.nlu
    assert actual.nuu  == expected.nuu
    assert actual.nslu == expected.nslu
    assert actual.nsuu == expected.nsuu

