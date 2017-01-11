# Optima

Optima is a C++ library for numerical solution of linear and nonlinear programing problems. Optima is mainly developed in C++ for performance reasons. A Python interface is available for a more convenient and simpler use of the scientific library. 

## Installation

In the steps below we will show how one can download Optima, build, and install it in Linux and MacOS systems. 

>**Note**: Compiling Optima can take some time. This is because it heavily relies on template metaprogramming for efficient vector and matrix calculations. Compilation of the Python wrappers can also take several minutes, as Boost.Python too relies on template metaprogramming.

### Downloading Optima
Optima source code is kept in this [Bitbucket repository](https://bitbucket.org/reaktoro/optima). If you have `git` installed in your system, then downloading this repository is as easy as running the following command in a terminal:

    git clone https://bitbucket.org/reaktoro/optima.git Optima

Alternatively, you can access this [link](https://bitbucket.org/reaktoro/optima/get/master.zip) to directly download Optima source code as a zipped file. If you choose this option, unzip the file before proceeding to the next step.

### Compiling the C++ library
Here we show how to compile only the C++ part of Optima. Its Python interface is an optional component of the project, and its compilation and installation is shown in the next section.

To build and install Optima, ensure that [CMake](https://cmake.org/) is installed in your system. Optima uses CMake for managing the whole build process, including the installation of third party libraries. 

Once CMake has been installed, go inside the directory of the downloaded Optima source code. In the terminal, execute the following commands:
    
    mkdir build && cd build
    cmake ..
    make -j3

The first line above creates a directory called `build` and changes the current directory to it in Linux and MacOS systems. The command `cmake ..`  tells CMake to configure the build process based on the main `CMakeLists.txt` file in the root directory of Optima's source code. Finally, `make -j3` compiles Optima's source code using 3 parallel processes. To use all available processors in your machine, execute `make -j` instead. 

>**Warning**: The use of all available resources for compiling Optima can freeze your machine.

To install the compiled libraries in your system, execute:
        
    make install

Note that this might require administrator rights, so that you would need to execute `sudo make install` instead. For a local installation, you can specify a directory path for the installed files using the CMake command:

    cmake .. -DCMAKE_INSTALL_PREFIX=/home/username/local/

### Compiling the Python interface
Most C++ classes and methods in Optima are accessible from Python. To use its Python interface, Python wrappers to these C++ components must be compiled. These wrappers are generated using [Boost.Python](http://www.boost.org/doc/libs/1_60_0/libs/python/doc/html/index.html), so ensure your system has Boost installed, including `libboost_python`.

To build the Python wrappers, the CMake option `-DBUILD_PYTHON=ON` must be provided to the CMake command configuring the build process:

    cmake .. -DBUILD_PYTHON=ON

## License

Optima is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Optima is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Optima. If not, see <http://www.gnu.org/licenses/>.

## Contact

For comments and requests, send an email to:

    allan.leal@erdw.ethz.ch