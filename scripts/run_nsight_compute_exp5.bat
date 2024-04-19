@echo off

set NCU_PATH="D:/Program Files/NVIDIA Corporation/Nsight Compute 2024.1.0/target/windows-desktop-win7-x64/ncu.exe"
set OUTPUT_DIR="./Experiment5"
set APP_EXE="./Experiment5.exe"
set METRICS=sm_efficiency,achieved_occupancy
set SECTIONS=MemoryWorkloadAnalysis
set LAUNCH_SKIP=2
set LAUNCH_COUNT=10

if not exist %APP_EXE% (
    echo Application executable not found: %APP_EXE%
    exit /b
)

%NCU_PATH% --config-file off --export %OUTPUT_DIR% --force-overwrite --set full %APP_EXE% --metrics %METRICS% --section %SECTIONS% --source-level-analysis --launch-skip %LAUNCH_SKIP% --launch-count %LAUNCH_COUNT%

echo Profiling Complete.
