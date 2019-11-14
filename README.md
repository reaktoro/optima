# Optima

## Overview

Optima is a C++ library for numerical solution of linear and nonlinear
programing problems. Optima is mainly developed in C++ for performance reasons.
A Python interface is available for a more convenient and simpler use of the
scientific library.

Optima is still under development, and its API can change at any time. The
algorithms are mature enough, but the way the optimization problems are defined
will most likely change (and be simplified).

## Installation

~~~
git clone https://github.com/reaktoro/optima
cd optima
mkdir build && cd build
cmake ..
make -j 3
~~~

This will create a `build` directory and install the library in
`build/release/install`.

## Testing

Execute `pytest .` in the root directory. Make sure `PYTHONPATH` contains the
path to `optima` Python module:

~~~
export PYTHONPATH=path/to/optima/build/release/install/lib
~~~

**More details will be added later.**
