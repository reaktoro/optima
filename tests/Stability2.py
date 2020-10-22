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
from numpy.testing import assert_almost_equal
from pytest import approx, mark

tested_nx  = [10, 20]  # The tested number of x variables
tested_np  = [0, 5]    # The tested number of p variables
tested_ny  = [5, 10]   # The tested number of y variables
tested_nz  = [0, 5]    # The tested number of z variables
tested_nl  = [0, 2]    # The tested number of linearly dependent rows in Ax
tested_nlu = [0, 2]    # The tested number of lower unstable variables
tested_nuu = [0, 2]    # The tested number of upper unstable variables

@mark.parametrize("nx",  tested_nx)
@mark.parametrize("np",  tested_np)
@mark.parametrize("ny",  tested_ny)
@mark.parametrize("nz",  tested_nz)
@mark.parametrize("nl",  tested_nl)
@mark.parametrize("nlu",  tested_nlu)
@mark.parametrize("nuu",  tested_nuu)
def test_stability(nx, np, ny, nz, nl, nlu, nuu):

    # Ensure nx is larger than np and (ny + nz)
    if nx < np or nx < ny + nz: return

    # Ensure nl < ny
    if ny < nl: return

    Ax = random.rand(ny, nx)
    Ap = random.rand(ny, np)

    Ax[:nl, :] = 0.0  # set last nl rows to be zero so that we have nl linearly dependent rows in Ax

    W = JacobianBlockW(nx, np, ny, nz, Ax, Ap)

    weights = ones(nx)
    Jx = random.rand(nz, nx)
    Jp = random.rand(nz, np)

    W.update(Jx, Jp, weights)

    Wbar = W.canonicalForm()

    jb = Wbar.jb
    jn = Wbar.jn
    R  = Wbar.R

    nb = len(jb)
    nn = len(jn)

    jlu = jn[:nlu]       # the first nlu non-basic variables are lower unstable
    juu = jn[nn - nuu:]  # the last  nuu non-basic variables are lower unstable

    juu = list(set(juu) - set(jlu))  # ensure no upper unstable variable is also lower unstable!

    x = abs(random.rand(nx))  # abs is used only to simplify the logic in determining xlower and xupper below

    xlower = abs(random.rand(nx)) - x
    xupper = abs(random.rand(nx)) + x

    xlower[jlu] = x[jlu]  # attach the lower unstable variables to their lower bound
    xupper[juu] = x[juu]  # attach the upper unstable variables to their upper bound

    Rb = R[:nb, :]
    Wn = W.Wx[:, jn]

    #==========================================================================
    # Initialize expected s = g - tr(Wx)*λ considering the unstable variables
    #==========================================================================
    g = zeros(nx)

    g[jb] = random.rand(nb)
    lmbda = Rb.T @ g[jb]

    s = random.rand(nx)

    s[jlu] =  1.0  # ensure s[i] > 0 for lower unstable variables
    s[juu] = -1.0  # ensure s[i] < 0 for upper unstable variables
    s[jb] = 0.0    # ensure s[i] = 0 for basic variables

    g[jn] = s[jn] + Wn.T @ lmbda

    #==========================================================================
    # Initialize Stability object to compute λ, s, js, ju = (jlu, juu)
    #==========================================================================

    stability = Stability2(nx)
    stability.update(W, g, x, xlower, xupper)

    status = stability.status()

    #==========================================================================
    # Check both λ = tr(R)*gb and stabilities s = g - tr(Wx)*λ
    #==========================================================================

    assert_almost_equal(status.lmbda, lmbda)
    assert_almost_equal(status.s, s)

    #==========================================================================
    # Check the indices of stable, lower unstable and upper unstable
    #==========================================================================

    ju = set(jlu) | set(juu)
    js = set(range(nx)) - ju

    assert set(status.jlu) == set(jlu)
    assert set(status.juu) == set(juu)
    assert set(status.ju) == set(ju)
    assert set(status.js) == set(js)
