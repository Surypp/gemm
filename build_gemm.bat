@echo off
setlocal

:: Usage: build_gemm.bat [ARCH_LIST] [BUILD_TYPE] [--clean]
::   ARCH_LIST defaults to "80;90a;120" (all three targets)
::   BUILD_TYPE defaults to Release
::   --clean  : wipe entire build/ directory before configuring
::
:: Single-arch examples:
::   build_gemm.bat 120
::   build_gemm.bat "80;120" Debug
::   build_gemm.bat 120 Release --clean

set ARCH_LIST=%~1
if "%ARCH_LIST%"=="" set ARCH_LIST=80;90a;120

set BUILD_TYPE=%~2
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

set DO_CLEAN=0
if "%~3"=="--clean" set DO_CLEAN=1

set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"

call %VCVARS% x64
if errorlevel 1 (
    echo [ERROR] Visual Studio environment not found
    exit /b 1
)

if "%DO_CLEAN%"=="1" (
    echo [clean] wiping build\ ...
    rmdir /s /q build 2>nul
) else if exist build\CMakeCache.txt (
    findstr /c:"CMAKE_GENERATOR:INTERNAL=Ninja" build\CMakeCache.txt >nul 2>&1
    if errorlevel 1 (
        echo [clean] non-Ninja cache detected, wiping CMakeCache + CMakeFiles
        del /f /q build\CMakeCache.txt
        rmdir /s /q build\CMakeFiles 2>nul
    )
)

cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DGEMM_ARCH_LIST="%ARCH_LIST%" ^
    -Wno-dev
if errorlevel 1 exit /b 1

cmake --build build -j8
if errorlevel 1 exit /b 1

echo.
echo Build OK -- arch: %ARCH_LIST%, type: %BUILD_TYPE%
