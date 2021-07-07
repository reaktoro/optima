@REM This file will be copied by cmake to the root of the build directory.
@REM Execute `envs4release` from that directory and the environment variables
@REM below will be update so that the python package and C++ libraries can
@REM be found if necessary (e.g., to execute pytest).

if not defined OPTIMA_BACKUP_PYTHONPATH set OPTIMA_BACKUP_PYTHONPATH=%PYTHONPATH%
if not defined OPTIMA_BACKUP_PATH set OPTIMA_BACKUP_PATH=%PATH%

set PYTHONPATH=%CD%\python\package\installed\Release\Lib\site-packages;%CD%\python\package\installed\Lib\site-packages;%OPTIMA_BACKUP_PYTHONPATH%
set PATH=%CD%\Optima\Release;%CD%\Optima;%OPTIMA_BACKUP_PATH%
