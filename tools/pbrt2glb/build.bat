@echo off
setlocal

REM Build pbrt2glb.exe as a single-file Windows executable using PyInstaller.

cd /d "%~dp0"

where py >nul 2>&1
if errorlevel 1 (
    echo [pbrt2glb] Python launcher 'py' not found. Install Python 3 first.
    exit /b 1
)

echo [pbrt2glb] Installing/updating dependencies...
py -3 -m pip install --upgrade pip
py -3 -m pip install -r requirements.txt
py -3 -m pip install pyinstaller
if errorlevel 1 goto :fail

echo [pbrt2glb] Running PyInstaller...
py -3 -m PyInstaller ^
    --onefile ^
    --name pbrt2glb ^
    --console ^
    --collect-submodules pygltflib ^
    --collect-submodules plyfile ^
    --collect-submodules imageio ^
    --hidden-import PIL ^
    pbrt2glb.py
if errorlevel 1 goto :fail

echo.
echo [pbrt2glb] Build complete. Output: %~dp0dist\pbrt2glb.exe
exit /b 0

:fail
echo.
echo [pbrt2glb] Build failed.
exit /b 1
