# How to execute the tests in Optima

Once the library and the Python bindings have been built, update `PYTHONPATH`
so that it knows the path to `optima`, the Optima package for Python:

~~~bash
export PYTHONPATH=$HOME/codes/optima/build/release/python
~~~

Now, from the root directory of the project, execute:

~~~bash
pytest .
~~~
