@echo off
setlocal EnableDelayedExpansion
rem SwarmUI runs this for the comfy-rust Self-Starting backend, and adds the port and model paths as arguments.
rem It doesn't build, so SwarmUI isn't held up: build first, or use start-backend.bat.
cd /D "%~dp0"

set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
set "SERVER=target\release\comfy-server.exe"

if not exist "%SERVER%" (
    echo %SERVER% is missing. Build it with: cargo build --release -p comfy-server --features native
    exit /b 1
)

rem A stray quote in PATH hides every later entry from cmd, so drop quotes here.
set "PATH=%CUDA_BIN%;!PATH:"=!"
"%SERVER%" --engine native %*
