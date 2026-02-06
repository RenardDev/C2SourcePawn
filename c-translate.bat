@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul

set "PY=py"
set "CLANG=C:\Program Files\LLVM\bin\clang.exe"
set "INCLUDE_DIR=%SCRIPT_DIR%include"
set "SLIM_H=%SCRIPT_DIR%slim.h"
set "GENSLIM_BAT=%SCRIPT_DIR%c-genslim.bat"
set "TRANSLATOR=%SCRIPT_DIR%translator.py"

set "SPCOMP=%SCRIPT_DIR%spcomp.exe"
if not exist "%SPCOMP%" set "SPCOMP=%SCRIPT_DIR%spcomp"
if not exist "%SPCOMP%" set "SPCOMP=spcomp"

set "TRANSLATE_DIR=%SCRIPT_DIR%translate"
set "KEEP_TEMP=0"
set "RUN_GENSLIM=1"

set "INPUT="
set "OUTPUT="

set "EXTRA_TRANSLATOR_ARGS="
set "EXTRA_SPCOMP_ARGS="

:parse
if "%~1"=="" goto parsed

if /I "%~1"=="-h" goto help
if /I "%~1"=="--help" goto help

if /I "%~1"=="-i" ( set "INPUT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--input" ( set "INPUT=%~2" & shift & shift & goto parse )

if /I "%~1"=="-o" ( set "OUTPUT=%~2" & shift & shift & goto parse )
if /I "%~1"=="--output" ( set "OUTPUT=%~2" & shift & shift & goto parse )

if /I "%~1"=="--clang" ( set "CLANG=%~2" & shift & shift & goto parse )
if /I "%~1"=="--include-dir" ( set "INCLUDE_DIR=%~2" & shift & shift & goto parse )
if /I "%~1"=="--slim" ( set "SLIM_H=%~2" & shift & shift & goto parse )
if /I "%~1"=="--spcomp" ( set "SPCOMP=%~2" & shift & shift & goto parse )

if /I "%~1"=="--translate-dir" ( set "TRANSLATE_DIR=%~2" & shift & shift & goto parse )
if /I "%~1"=="--build-dir" ( set "TRANSLATE_DIR=%~2" & shift & shift & goto parse )

if /I "%~1"=="--no-genslim" ( set "RUN_GENSLIM=0" & shift & goto parse )
if /I "%~1"=="--keep-temp" ( set "KEEP_TEMP=1" & shift & goto parse )

set "EXTRA_TRANSLATOR_ARGS=!EXTRA_TRANSLATOR_ARGS! %~1"
shift
goto parse

:parsed
if "%INPUT%"=="" (
  echo [ERR] Input not specified. Use: c-translate.bat -i input.c [-o output.sp]
  goto help
)

if not exist "%INPUT%" (
  echo [ERR] Input file not found: "%INPUT%"
  goto fail
)

if "%OUTPUT%"=="" (
  for %%F in ("%INPUT%") do set "OUTPUT=%%~dpnF.sp"
)

if not exist "%TRANSLATE_DIR%" mkdir "%TRANSLATE_DIR%" >nul 2>&1

set "LOG_PASS1=%TRANSLATE_DIR%\spcomp_pass1.txt"
set "LOG_FINAL=%TRANSLATE_DIR%\spcomp_final.txt"
set "IGNORE_FILE=%TRANSLATE_DIR%\ignore_symbols.txt"

echo ============================================================
echo INPUT        : "%INPUT%"
echo OUTPUT       : "%OUTPUT%"
echo CLANG        : "%CLANG%"
echo SLIM_H       : "%SLIM_H%"
echo INCLUDEDIR   : "%INCLUDE_DIR%"
echo SPCOMP       : "%SPCOMP%"
echo TRANSLATEDIR : "%TRANSLATE_DIR%"
echo GENSLIM      : "%RUN_GENSLIM%"
echo ============================================================

if not exist "%CLANG%" (
  echo [ERR] clang not found: "%CLANG%"
  goto fail
)

if not exist "%GENSLIM_BAT%" (
  echo [ERR] c-genslim.bat not found: "%GENSLIM_BAT%"
  goto fail
)

if not exist "%TRANSLATOR%" (
  echo [ERR] translator.py not found: "%TRANSLATOR%"
  goto fail
)

if "%RUN_GENSLIM%"=="1" (
  if not exist "%INCLUDE_DIR%" (
    echo [ERR] include dir not found: "%INCLUDE_DIR%"
    goto fail
  )
  echo [1/5] c-genslim.bat "%INCLUDE_DIR%" "%SLIM_H%"
  call "%GENSLIM_BAT%" -i "%INCLUDE_DIR%" -o "%SLIM_H%"
  if errorlevel 1 (
    echo [ERR] c-genslim failed
    goto fail
  )
) else (
  echo [1/5] genslim skipped (--no-genslim)
)

if not exist "%SLIM_H%" (
  echo [ERR] slim header not found after genslim: "%SLIM_H%"
  goto fail
)

echo [2/5] translator pass1
"%PY%" "%TRANSLATOR%" -o "%OUTPUT%" "%INPUT%" --clang "%CLANG%" --clang-arg=-include --clang-arg="%SLIM_H%" %EXTRA_TRANSLATOR_ARGS%
if errorlevel 1 (
  echo [ERR] translator pass1 failed
  goto fail
)

echo [3/5] spcomp pass1 ^> "%LOG_PASS1%"
"%SPCOMP%" "%OUTPUT%" %EXTRA_SPCOMP_ARGS% > "%LOG_PASS1%" 2>&1
type "%LOG_PASS1%" > "%IGNORE_FILE%"

echo [4/5] translator pass2 --ignore-symbols="%IGNORE_FILE%"
"%PY%" "%TRANSLATOR%" -o "%OUTPUT%" "%INPUT%" --clang "%CLANG%" --clang-arg=-include --clang-arg="%SLIM_H%" --ignore-symbols="%IGNORE_FILE%" %EXTRA_TRANSLATOR_ARGS%
if errorlevel 1 (
  echo [ERR] translator pass2 failed
  goto fail
)

echo [5/5] spcomp final ^> "%LOG_FINAL%"
"%SPCOMP%" "%OUTPUT%" %EXTRA_SPCOMP_ARGS% > "%LOG_FINAL%" 2>&1
set "SPCOMP_EXIT=%ERRORLEVEL%"

if "%KEEP_TEMP%"=="0" (
  del /q "%IGNORE_FILE%" >nul 2>&1
)

echo ============================================================
echo Done. Output: "%OUTPUT%"
echo Pass1 log  : "%LOG_PASS1%"
echo Final log  : "%LOG_FINAL%"
echo ============================================================

popd >nul
exit /b %SPCOMP_EXIT%

:help
echo.
echo Usage:
echo   c-translate.bat -i input.c [-o output.sp] [options] [extra translator args...]
echo.
echo Options:
echo   --clang "C:\Program Files\LLVM\bin\clang.exe"
echo   --slim  ".\slim.h"
echo   --include-dir ".\include"
echo   --spcomp ".\spcomp.exe"
echo   --translate-dir ".\translate"   ^(alias: --build-dir^)
echo   --no-genslim
echo   --keep-temp
echo.
echo Examples:
echo   c-translate.bat -i demo.c -o demo.sp
echo   c-translate.bat -i demo.c -- --strict --std c99
echo.
popd >nul
exit /b 0

:fail
popd >nul
exit /b 1
