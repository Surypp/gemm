@echo off
setlocal

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
        del /f /q build\CMakeCache.txt
        rmdir /s /q build\CMakeFiles 2>nul
    )
)

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
