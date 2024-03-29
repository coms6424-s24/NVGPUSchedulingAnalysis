@echo off

set VSCOMPILER="E:\visual studio\VC\Tools\MSVC\14.39.33519\bin\Hostx64\x64"
set SRCFILES=../src/Experiment1.cu ../src/KernelsExp1.cu ../src/TimerSpin.cu
set OUTPUT=-o Experiment1.exe
set INCLUDES=-I "../include"
set DEBUGFLAGS=-G -lineinfo -Xptxas -v

nvcc -ccbin %VSCOMPILER% %SRCFILES% %OUTPUT% %DEBUGFLAGS% %INCLUDES% %LIBPATHS% %ARCH_FLAGS% -lcudart -std=c++17

echo Compilation complete.