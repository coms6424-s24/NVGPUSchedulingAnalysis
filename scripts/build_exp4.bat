@echo off

set VSCOMPILER="D:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.34.31933/bin/Hostx64/x64"
set SRCFILES=../src/Experiment4.cu ../src/KernelsExp4.cu
set OUTPUT=-o Experiment4.exe
set INCLUDES=-I "../include"
set DEBUGFLAGS=-G -lineinfo -Xptxas -v
set RELEASEFLAGS=-lineinfo -Xptxas -v -O3

nvcc -ccbin %VSCOMPILER% %SRCFILES% %OUTPUT% %DEBUGFLAGS% %INCLUDES% %LIBPATHS% %ARCH_FLAGS% -lcudart -std=c++17

echo Compilation complete.