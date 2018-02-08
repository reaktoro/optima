from optima import *
from numpy import *
from pytest import approx

def head(sequence, n):
    return sequence[:n]
    

def tail(sequence, n):
    return sequence[-n:]


def check_canonical_form(canonicalizer):
    R = canonicalizer.R()
    Q = canonicalizer.Q()
    C = canonicalizer.C()
    assert R.dot(A[:,Q]) == approx(C)


def test_canonicalizer_simple_case():
    m = 4
    n = 10
    A = random.rand(m, n)
    canonicalizer = Canonicalizer(A)
    r = canonicalizer.numBasicVariables()
    
    # Check the computed canonical form
    check_canonical_form(canonicalizer)

    # Perform a series of basis swap operations and check the canonical form
    for i in xrange(r):
        for j in xrange(n - r):
            canonicalizer.swapBasicVariable(i, j)
            check_canonical_form(canonicalizer)

    # Change the order of the variables and see if the updated canonical form is correct
    ordering = arange(n)
    head(ordering, m) = linspace(n - 1, n - m, m)
    tail(ordering, n - m) = linspace(0, n - m, n - m)

    print ordering