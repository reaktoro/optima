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
from pytest import approx, mark
from itertools import product

# The number of *variables* in *x*.
n = 20

# The number of *lower unstable variables* in *x*.
tested_nlu = [0, 1]

# The number of *upper unstable variables* in *x*.
tested_nuu = [0, 3]

# The number of *strictly lower unstable variables* in *x*.
tested_nslu = [0, 5]

# The number of *strictly upper unstable variables* in *x*.
tested_nsuu = [0, 7]

# Combination of all tested cases
testdata = product(
    tested_nlu,
    tested_nuu,
    tested_nslu,
    tested_nsuu
)

@mark.parametrize("args", testdata)
def test_stability(args):

    nlu, nuu, nslu, nsuu = args

    ns = n - (nlu + nuu + nslu + nsuu)

    data = StabilityData()
    data.iordering = arange(n)
    data.ns = ns
    data.nlu = nlu
    data.nuu = nuu
    data.nslu = nslu
    data.nsuu = nsuu

    def check():
        jx = data.iordering      # the indices of the variables
        js = jx[:ns]             # the indices of the stable
        ju = jx[ns:]             # the indices of the unstable variables
        jlu = ju[:nlu]           # the indices of the lower unstable variables
        juu = ju[nlu:nlu + nuu]  # the indices of the upper unstable variables
        jsu = ju[nlu + nuu:]     # the indices of the strictly lower/upper unstable variables
        jslu = jsu[:nslu]        # the indices of the strictly lower unstable variables
        jsuu = jsu[nslu:]        # the indices of the strictly upper unstable variables

        assert stability.numVariables() == n
        assert stability.numStableVariables() == ns
        assert stability.numUnstableVariables() == n - ns
        assert stability.numLowerUnstableVariables() == nlu
        assert stability.numUpperUnstableVariables() == nuu
        assert stability.numStrictlyLowerUnstableVariables() == nslu
        assert stability.numStrictlyUpperUnstableVariables() == nsuu
        assert stability.numStrictlyUnstableVariables() == nslu + nsuu
        assert (stability.indicesVariables() == jx).all()
        assert (stability.indicesStableVariables() == js).all()
        assert (stability.indicesUnstableVariables() == ju).all()
        assert (stability.indicesLowerUnstableVariables() == jlu).all()
        assert (stability.indicesUpperUnstableVariables() == juu).all()
        assert (stability.indicesStrictlyLowerUnstableVariables() == jslu).all()
        assert (stability.indicesStrictlyUpperUnstableVariables() == jsuu).all()
        assert (stability.indicesStrictlyUnstableVariables() == jsu).all()

    # Check constructor
    stability = Stability(data)

    check()

    # Check update method
    stability = Stability()
    stability.update(data)

    check()
