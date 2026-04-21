@echo off
setlocal

<<<<<<< HEAD
echo ========================================
echo  GEMM Build Script
echo ========================================
echo.

:: Configuration
set PROJECT_DIR=%~dp0
set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"
set ARCH=x64
set BUILD_TYPE=Release
set GEMM_ARCH=120
set JOBS=8

:: Aller dans le dossier projet
cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo [ERREUR] Dossier introuvable : %PROJECT_DIR%
    pause
    exit /b 1
)

:: Charger l'environnement Visual Studio
call %VCVARS% %ARCH%
if errorlevel 1 (
    echo [ERREUR] Impossible de charger Visual Studio
    pause
    exit /b 1
)

:: Supprimer le cache CMake si le generateur est different (NMake → Ninja)
if exist build\CMakeCache.txt (
    findstr /c:"CMAKE_GENERATOR:INTERNAL=Ninja" build\CMakeCache.txt >nul 2>&1
    if errorlevel 1 (
        echo [INFO] Cache CMake avec mauvais generateur detecte, suppression...
=======
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

set VCVARS="C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
if not exist %VCVARS% set VCVARS="C:\Program Files\Microsoft Visual Studio\17\Community\VC\Auxiliary\Build\vcvarsall.bat"
if not exist %VCVARS% set VCVARS="C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat"

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
>>>>>>> 1572328 (refactor: restructured phase directories)
        del /f /q build\CMakeCache.txt
        rmdir /s /q build\CMakeFiles 2>nul
    )
)

<<<<<<< HEAD
echo.
echo [1/2] Configuration CMake avec Ninja...
echo ----------------------------------------
cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DGEMM_ARCH_LIST=%GEMM_ARCH% ^
    -Wno-dev
if errorlevel 1 (
    echo [ERREUR] Configuration CMake echouee
    pause
    exit /b 1
)

echo.
echo [2/2] Compilation en cours...
echo ----------------------------------------
cmake --build build -j%JOBS%
if errorlevel 1 (
    echo [ERREUR] Compilation echouee
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Build termine avec succes !
echo ========================================
pause
=======
cmake -B build -G Ninja ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DGEMM_ARCH_LIST="%ARCH_LIST%" ^
    -Wno-dev
if errorlevel 1 exit /b 1

cmake --build build -j8
if errorlevel 1 exit /b 1

echo.
echo Build OK -- arch: %ARCH_LIST%, type: %BUILD_TYPE%
>>>>>>> 1572328 (refactor: restructured phase directories)
