@echo off

set VSCOMPILER="D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64"
set SRCFILES=../src/Experiment1.cu ../src/Kernels.cu ../src/TimerSpin.cu
set OUTPUT=-o Experiment1.exe
set INCLUDES=-I "../include"
set DEBUGFLAGS=-G -lineinfo -Xptxas -v

nvcc -ccbin %VSCOMPILER% %SRCFILES% %OUTPUT% %DEBUGFLAGS% %INCLUDES% %LIBPATHS% %ARCH_FLAGS% -lcudart -std=c++17

echo Compilation complete.