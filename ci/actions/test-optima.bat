echo --- current directory: %cd% ---

REM Activate the conda environment
set PATH=%CONDA%;%CONDA%\Scripts;%CONDA%\Library\bin;%PATH% || goto :error
call activate optima || goto :error

echo === Running tests... || goto :error
cmake --build build --config %CONFIGURATION% --target tests || goto :error
echo === Running tests...finished! || goto :error

exit /B 0

:error
echo ERROR!
echo Command exit status: %ERRORLEVEL%
