from optima import *
from numpy import *
from pytest import approx

def reverse(list):
    return list[::-1]


def check_canonical_form(canonicalizer, A):
    R = canonicalizer.R()
    Q = canonicalizer.Q()
    C = canonicalizer.C()
    assert R.dot(A[:,Q]) == approx(C)


def check_canonical_ordering(canonicalizer, weigths):
    n = canonicalizer.numVariables()
    nb = canonicalizer.numBasicVariables()
    nn = canonicalizer.numNonBasicVariables()
    ibasic = canonicalizer.indicesBasicVariables()
    inonbasic = canonicalizer.indicesNonBasicVariables()
    for i in range(1, nb):
        assert weigths[ibasic[i]] <= weigths[ibasic[i - 1]]
    for i in range(1, nn):
        assert weigths[inonbasic[i]] <= weigths[inonbasic[i - 1]]


def check_canonicalizer(canonicalizer, A):
    # Auxiliary variables
    n = canonicalizer.numVariables()
    m = canonicalizer.numEquations()
    nb = canonicalizer.numBasicVariables()

    #---------------------------------------------------------------------------
    # Check the computed canonical form
    #---------------------------------------------------------------------------
    check_canonical_form(canonicalizer, A)

    #---------------------------------------------------------------------------
    # Perform a series of basis swap operations and check the canonical form
    #---------------------------------------------------------------------------
    for i in xrange(nb):
        for j in xrange(n - nb):
            canonicalizer.updateWithSwapBasicVariable(i, j)
            check_canonical_form(canonicalizer, A)

    #---------------------------------------------------------------------------
    # Change the order of the variables and see if the updated canonical
    # form is correct
    #---------------------------------------------------------------------------
    ordering = reverse(range(n - m, n)) + range(n - m)

    canonicalizer.updateWithNewOrdering(ordering)

    A = A[:, ordering]  # Reorder the columns of matrix A

    check_canonical_form(canonicalizer, A)

    #---------------------------------------------------------------------------
    # Set weights for the variables to update the basic/non-basic partition
    #---------------------------------------------------------------------------
    weigths = abs(random.rand(n)) + 1.0

    canonicalizer.updateWithPriorityWeights(weigths)

    check_canonical_form(canonicalizer, A)

    check_canonical_ordering(canonicalizer, weigths)


def test_canonicalizer_with_regular_matrix():
    m = 4
    n = 10
    A = random.rand(m, n)
    canonicalizer = Canonicalizer(A)
    check_canonicalizer(canonicalizer, A)


def test_canonicalizer_with_two_linearly_dependent_rows():
    m = 4
    n = 10
    A = random.rand(m, n)
    A[2] = A[0] + 2*A[1]  # row(2) = row(0) + 2*row(1)
    A[3] = A[1] - 2*A[2]  # row(3) = row(1) - 2*row(2)
    canonicalizer = Canonicalizer(A)
    check_canonicalizer(canonicalizer, A)


def test_canonicalizer_with_fixed_variables():
    m = 4
    n = 10
    A = random.rand(m, n)
    A[2] = A[0] + 2*A[1]  # row(2) = row(0) + 2*row(1)
    A[3] = A[1] - 2*A[2]  # row(3) = row(1) - 2*row(2)
    canonicalizer = Canonicalizer(A)
    check_canonicalizer(canonicalizer, A)
    
    
def test_canonicalizer_with_rational_numbers():
    m = 4
    n = 10
    maxdenominator = 10
    A = (arange(m*n) / arange(10, m*n + 10)).reshape((m, n)) 
    canonicalizer = Canonicalizer(A)
    canonicalizer.rationalize(100) 
    check_canonicalizer(canonicalizer, A)
    
    
    
