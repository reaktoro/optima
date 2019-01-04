# Optima

## Overview

Optima is a C++ library for numerical solution of linear and nonlinear programing problems. Optima is mainly developed in C++ for performance reasons. A Python interface is available for a more convenient and simpler use of the scientific library. 

## Installation

~~~
git clone https://github.com/reaktoro/optima
cd optima
cmake -P install
~~~

This will create a `build` directory and install the library in `build/release/install`. 

## Testing

Execute `pytest .` under `optima/tests`. Make sure `PYTHONPATH` contains path to `optima` Python module:

~~~
export PYTHONPATH=path/to/optima/build/release/install/lib
~~~

**More details will be added later.**
