@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cd /d C:\Users\lance\Music\gemm\build
ninja gemm_tests -j8
echo Exit code: %ERRORLEVEL%
