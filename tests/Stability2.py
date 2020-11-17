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


from testing.optima import *
from testing.utils.matrices import *


tested_nx      = [15, 20]       # The tested number of x variables
tested_np      = [0, 5]         # The tested number of p variables
tested_ny      = [2, 5]         # The tested number of y variables
tested_nz      = [0, 5]         # The tested number of z variables
tested_nl      = [0, 2]         # The tested number of linearly dependent rows in Ax
tested_nlu     = [0, 2]         # The tested number of lower unstable variables
tested_nuu     = [0, 2]         # The tested number of upper unstable variables
tested_diagHxx = [False, True]  # The tested options for Hxx structure

@pytest.mark.parametrize("nx"     , tested_nx)
@pytest.mark.parametrize("np"     , tested_np)
@pytest.mark.parametrize("ny"     , tested_ny)
@pytest.mark.parametrize("nz"     , tested_nz)
@pytest.mark.parametrize("nl"     , tested_nl)
@pytest.mark.parametrize("nlu"    , tested_nlu)
@pytest.mark.parametrize("nuu"    , tested_nuu)
@pytest.mark.parametrize("diagHxx", tested_diagHxx)
def testStability(nx, np, ny, nz, nl, nlu, nuu, diagHxx):

    nu = nlu + nuu

    params = MasterParams(nx, np, ny, nz, nl, nu, diagHxx)

    if params.invalid(): return

    RWQ = createMatrixRWQ(params)

    W    = RWQ.asMatrixViewW()
    Wbar = RWQ.asMatrixViewRWQ()

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
    g = npy.zeros(nx)

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
    stability.update(RWQ, g, x, xlower, xupper)

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
