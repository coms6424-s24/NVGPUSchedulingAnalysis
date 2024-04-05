@echo off

set VSCOMPILER="E:\visual studio\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
set SRCFILES=../src/Experiment2.cu ../src/KernelsExp1.cu ../src/TimerSpin.cu
set OUTPUT=-o Experiment2.exe
set INCLUDES=-I "../include"
set DEBUGFLAGS=-G -lineinfo -Xptxas -v
set RELEASEFLAGS=-lineinfo -Xptxas -v -O3

nvcc -ccbin %VSCOMPILER% %SRCFILES% %OUTPUT% %RELEASEFLAGS% %INCLUDES% %LIBPATHS% %ARCH_FLAGS% -lcudart -std=c++17

echo Compilation complete.