# How to execute the tests in Optima

Once the library and the Python bindings have been built, Optima should be
tested.

## Linux or macOS

In the build directory, execute the following:

~~~text
make tests
~~~

## Windows

In the build directory, execute the following if Ninja was used as CMake generator:

~~~text
ninja tests
~~~

Alternatively, execute the following which works for any generator:

~~~text
cmake --build . --target tests
~~~
