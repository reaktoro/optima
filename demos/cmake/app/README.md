Using CMake to Build a C++ Optima App
=====================================

This example demonstrates how your C++ Optima application (executable) can be
built using [CMake].

In the root directory of this example, execute the following in the
terminal: :

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

## Using a custom installation of Optima

If you have a custom installation of Optima (i.e., Optima is not in a path that
CMake can find it), you'll need to use the following (instead of just `cmake
..` as shown above): :

    cmake .. -DCMAKE_PREFIX_PATH=/path/to/optima/installation

After building the application, you can run the app:

    $ ./app

which should output the result of a simple calculation.

## Using Conda

If you are using Conda to manage the dependencies of Optima, then you
should activate the Optima conda environment:

    conda activate optima

Read more on the installation instructions of Optima.

  [CMake]: https://cmake.org/