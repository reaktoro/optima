@REM This file will be copied by cmake to the root of the build directory.
@REM Execute `envs4debug` from that directory and the environment variables
@REM below will be update so that the python package and C++ libraries can
@REM be found if necessary (e.g., to execute pytest).

set PYTHONPATH=%CD%\python\package\installed\Debug\Lib\site-packages;%PYTHONPATH%
set PATH=%CD%\Optima\Debug;%PATH%

echo The following environment variables have been updated:
echo   PYTHONPATH = %PYTHONPATH%
echo   PATH = %PATH%
