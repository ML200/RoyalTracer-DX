# Setup the project
## Prerequisites:
- Install the Windows 10 SDK version 10.0.22621.0 using the VS installer
- Install Visual Studio 2022

## Clion: 2024.3.2 or newer
- Set up the toolchain: Visual Studio (should be auto-detected, select it). Delete any other toolchain.
- Configure the CMAKE project: Select Visual Studio as the toolchain. The build directory should be named "cmake-build-debug-visual-studio" to exempt it from pushing to GitHub. Select "use default" for the generator and "Debug" for build type.
- Delete the current cmake-build-debug-visual-studio directory if it exists
- Reload the CMAKE project (file -> reload CMAKE project)
- Build and run the project. Includes should be automatically included in the build directory.

## CMAKE links:
- The Windows SDK should be located at "C:/Program Files (x86)/Windows Kits/10"
