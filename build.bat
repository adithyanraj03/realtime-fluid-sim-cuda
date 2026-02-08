@echo off
REM Navier-Stokes Fluid Simulator — Build Script (Windows / MSVC + CUDA + Ninja)

REM locate VS dev environment
set "VSDEVCMD="
for %%V in (18 2022 2019) do (
    for %%E in (Community Professional Enterprise) do (
        if exist "C:\Program Files\Microsoft Visual Studio\%%V\%%E\Common7\Tools\VsDevCmd.bat" (
            set "VSDEVCMD=C:\Program Files\Microsoft Visual Studio\%%V\%%E\Common7\Tools\VsDevCmd.bat"
            goto :found
        )
    )
)
:found
if "%VSDEVCMD%"=="" (
    echo ERROR: Visual Studio not found
    exit /b 1
)

echo Using VS: %VSDEVCMD%
call "%VSDEVCMD%" -arch=amd64 >nul 2>&1

REM locate cmake and ninja from VS
for /f "delims=" %%i in ('where cmake 2^>nul') do set "CMAKE=%%i"
if "%CMAKE%"=="" (
    for %%V in (18 2022 2019) do (
        for %%E in (Community Professional Enterprise) do (
            if exist "C:\Program Files\Microsoft Visual Studio\%%V\%%E\Common\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe" (
                set "CMAKE=C:\Program Files\Microsoft Visual Studio\%%V\%%E\Common\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
                goto :cmakefound
            )
        )
    )
)
:cmakefound

set "NINJA="
for %%V in (18 2022 2019) do (
    for %%E in (Community Professional Enterprise) do (
        if exist "C:\Program Files\Microsoft Visual Studio\%%V\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" (
            set "NINJA=C:\Program Files\Microsoft Visual Studio\%%V\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
            goto :ninjafound
        )
    )
)
:ninjafound

echo CMake: %CMAKE%
echo Ninja: %NINJA%

REM clean build directory
if exist build rmdir /s /q build
mkdir build
cd build

REM configure + build
"%CMAKE%" -G Ninja -DCMAKE_MAKE_PROGRAM="%NINJA%" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8/bin/nvcc.exe" -S .. -B .
if errorlevel 1 (
    echo CMake configure FAILED
    exit /b 1
)

"%CMAKE%" --build .
if errorlevel 1 (
    echo Build FAILED
    exit /b 1
)

echo Build SUCCEEDED
