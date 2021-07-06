# How to activate environment variables for Optima development and testing

In the build directory, execute the following to have the environment variables
`PYTHONPATH`, `LD_LIBRARY_PATH` and `PATH` set so that the python package
`optima` and the Optima C++ libraries can be found.

## Linux or macOS

~~~text
source envs
~~~

## Windows

For **Debug** development, execute:

~~~text
envs4debug.bat
~~~

For **Release** development, execute:

~~~text
envs4release.bat
~~~
