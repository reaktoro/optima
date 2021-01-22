echo --- current directory: %cd% ---

REM Activate the conda environment
set PATH=%CONDA%;%CONDA%\Scripts;%CONDA%\Library\bin;%PATH% || goto :error
call activate optima || goto :error

echo === Configuring Optima...
cmake -S . -B build -DCMAKE_INSTALL_PREFIX=install || goto :error
echo === Configuring Optima...finished!

echo === Building and installing Optima...
cmake --build build --config %CONFIGURATION% --target install || goto :error
echo === Building and installing Optima...finished!

exit /B 0

:error
echo ERROR!
echo Command exit status: %ERRORLEVEL%
