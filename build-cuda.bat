@echo off
REM Build script for CUDA version

REM Initialize MSVC environment
call "D:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvars64.bat"

REM Auto-detect CUDA compute capability from nvidia-smi
set "CUDA_COMPUTE_CAP="
for /f "skip=1 tokens=1 delims=." %%i in ('nvidia-smi --query-gpu=compute_cap --format=csv 2^>nul') do (
    set "CUDA_COMPUTE_CAP=%%i"
    goto :got_cc
)

:got_cc
if not defined CUDA_COMPUTE_CAP (
    echo [INFO] No NVIDIA GPU detected, using default compute capability 89
    set "CUDA_COMPUTE_CAP=89"
) else (
    echo [INFO] Detected CUDA compute capability: %CUDA_COMPUTE_CAP%
)

REM Build
cargo build --release --features cuda
cargo run --release --features cuda

echo.
echo Build complete! Binary at: target\release\hunyuan-infer.exe
