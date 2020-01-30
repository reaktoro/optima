Requirements
------------

Optima uses [CMake][1] for configuration and installation.

Building and Installing
-----------------------

To build and install Optima, issue the following commands from within its root directory:

    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    sudo make -j install
    
Alternatively, you can also install Optima running:
    
    cmake -P install

The previous steps for installing Optima should compile and install it in the default installation directory of your system (e.g, `/usr/local/` or `/opt/local/`).

Building Demos and Tests
------------------------

To build demos, enter the build directory (if any) and type:

    make demos

This will build all demos. To build a specific demo, just name the demo you want to build, for example building the demo `mydemo` requires:

    make mydemo

Similarly, to build the tests, enter the build directory (if any) and type:

    make tests

Next, you can execute all tests by issuing the command:

    make test

Alternatively, you may navigate to the tests directory in the build directory
and type:

    ctest

  [1]: http://www.cmake.org/
