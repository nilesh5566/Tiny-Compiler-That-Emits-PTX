@echo off
REM build.bat - configure, build and test KernC on Windows.
REM Run it from "Developer PowerShell for VS 2022" (or "x64 Native Tools Command Prompt")
REM so that CMake and the MSVC compiler are on PATH:   .\build.bat
setlocal
cd /d "%~dp0"

where cmake >nul 2>nul
if errorlevel 1 (
  echo [error] cmake not found. Open "Developer PowerShell for VS 2022" and run .\build.bat again.
  echo         See docs\INSTALL_WINDOWS.md, steps 2-3.
  exit /b 1
)

set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py

echo === [1/4] configure ===
cmake -S . -B build || goto :fail

echo === [2/4] build ===
cmake --build build --config Release || goto :fail

echo === [3/4] unit tests ===
build\bin\kernc_unit_tests.exe || goto :fail

echo === [4/4] verification suite ===
%PY% tools\run_tests.py || goto :fail

echo.
echo All done. Try:  build\bin\kernc.exe compile examples\saxpy.kern -O1 -o saxpy.ptx
echo                 %PY% tools\demo.py
exit /b 0

:fail
echo.
echo [error] a step failed - see the messages above and docs\INSTALL_WINDOWS.md (Troubleshooting).
exit /b 1
