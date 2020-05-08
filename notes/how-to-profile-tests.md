# How to profile the tests in Optima

Sometimes some tests can take relatively longer times to finish. Here are first steps to ensure fast test execution.

1. Ensure there is no hidden output (e.g. print statements) happening during
   the tests. Verify all tests do not output to the terminal by executing
   pytest with the `-s` flag. Allow output only when a failure is detected.

2. If test execution continues not fast enough, profile:
   ~~~
   python -m cProfile -o profile $(which pytest) tests/
   ~~~
   Then, open the output file `profile` with a profiling visualization tool, such as `snakeviz`:
   ~~~
   snakeviz profile
   ~~~

