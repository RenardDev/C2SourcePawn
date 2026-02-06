@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PY=py"
set "PARSER=%SCRIPT_DIR%parser.py"

set "INCLUDE_DIR="
set "OUT_FILE="

set "EXTRA_PARSER_ARGS="

:parse
if "%~1"=="" goto parsed

if /I "%~1"=="-h" goto help
if /I "%~1"=="--help" goto help

if /I "%~1"=="-i" ( set "INCLUDE_DIR=%~2" & shift & shift & goto parse )
if /I "%~1"=="--include-dir" ( set "INCLUDE_DIR=%~2" & shift & shift & goto parse )

if /I "%~1"=="-o" ( set "OUT_FILE=%~2" & shift & shift & goto parse )
if /I "%~1"=="--output" ( set "OUT_FILE=%~2" & shift & shift & goto parse )

set "EXTRA_PARSER_ARGS=!EXTRA_PARSER_ARGS! %~1"
shift
goto parse

:parsed
if "%INCLUDE_DIR%"=="" (
  echo [ERR] Include dir not specified. Use: c-genslim.bat -i .\include -o .\slim.h
  goto help
)

if "%OUT_FILE%"=="" (
  set "OUT_FILE=%SCRIPT_DIR%slim.h"
)

if not exist "%INCLUDE_DIR%" (
  echo [ERR] include dir not found: "%INCLUDE_DIR%"
  goto fail
)

if not exist "%PARSER%" (
  echo [ERR] parser.py not found: "%PARSER%"
  goto fail
)

echo ============================================================
echo INCLUDEDIR : "%INCLUDE_DIR%"
echo OUTPUT    : "%OUT_FILE%"
echo ============================================================

"%PY%" "%PARSER%" "%INCLUDE_DIR%" %EXTRA_PARSER_ARGS%
if errorlevel 1 (
  echo [ERR] parser failed
  goto fail
)

if not exist "%OUT_FILE%" (
  echo [WARN] "%OUT_FILE%" not found. parser.py may be writing to a fixed path.
  echo [WARN] If parser.py supports output arg, pass it via EXTRA args.
)

popd >nul
exit /b 0

:help
echo.
echo Usage:
echo   c-genslim.bat -i .\include [-o .\slim.h] [extra parser args...]
echo.
echo Examples:
echo   c-genslim.bat -i .\include -o .\slim.h
echo.
popd >nul
exit /b 0

:fail
popd >nul
exit /b 1
