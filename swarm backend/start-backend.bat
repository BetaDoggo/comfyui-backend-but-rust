@echo off
setlocal EnableDelayedExpansion
rem Starts the native SwarmUI backend. Pass extra comfy-server flags as arguments.
cd /D "%~dp0"

set "PORT=8188"
set "MODELS=%MODELS_ROOT%"
if not defined MODELS (
    echo Set MODELS_ROOT to your models folder first, for example:
    echo   set MODELS_ROOT=C:\SwarmUI\Models
    goto failed
)
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin"
set "SERVER=target\release\comfy-server.exe"

rem A stray quote in PATH hides every later entry from cmd, including cargo, so drop quotes here.
set "PATH=%CUDA_BIN%;!PATH:"=!"
set ROOTS=--models-root "%MODELS%"

rem cargo only rebuilds what changed, and the engine only exists in a --features native build.
echo Building (the first build takes about 10 minutes)...
cargo build --release -p comfy-server --features native
if errorlevel 1 goto failed

echo.
echo ================================================================
echo  native backend
echo.
echo  Connect SwarmUI:
echo    1. Server tab, Backends, add "ComfyUI API By URL"
echo    2. Address: http://127.0.0.1:%PORT%
echo    3. Pick the kroma W8A8 model, an int8 Anima model or an int8 MOSS Image model
echo    4. For turbo models, set CFG Scale 1 and Steps 8
echo.
echo  With ComfyUI as well, set Exact Backend ID to this backend's
echo  number in a preset to keep Krea 2 here.
echo.
echo  Models: %MODELS%
echo  Press Ctrl+C to stop.
echo ================================================================
echo.

"%SERVER%" --engine native --port %PORT% %ROOTS% %*
if errorlevel 1 goto failed
goto :eof

:failed
echo.
echo The native backend stopped with an error (see above).
pause
exit /b 1
