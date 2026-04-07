@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
cmake -B build -S . -DGEMM_ARCH_LIST=120 -DCMAKE_BUILD_TYPE=Release -DGEMM_ENABLE_HOPPER=OFF -DGEMM_ENABLE_TESTING=ON -DGEMM_ENABLE_BENCH=ON -G Ninja
