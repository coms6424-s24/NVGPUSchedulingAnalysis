@echo off

set VSCOMPILER="E:\visual studio\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
set SRCFILES=../src/TestingFramework.cu ../src/SchedulingPredictor.cpp ../src/KernelGenerator.cu
set OUTPUT=-o TestingFramework.exe
set INCLUDES=-I "../include"
set DEBUGFLAGS=-G -lineinfo -Xptxas -v,
set RELEASEFLAGS=-lineinfo -Xptxas -v, -O3

nvcc -ccbin %VSCOMPILER% %SRCFILES% %OUTPUT% %RELEASEFLAGS% %INCLUDES% %LIBPATHS% %ARCH_FLAGS% -lcudart -std=c++17

echo Compilation complete.