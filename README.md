

## Setting up the project
### Prerequisites:
- A reasonably recent Windows 11 version, as this project uses the DirectX Agility SDK
- Install Visual Studio 2022 (build tools)

### Clion: 2024.3.2 or newer
- Set up the toolchain: Visual Studio (should be auto-detected, select it). Delete any other toolchain.
- Configure the CMAKE project: Select Visual Studio as the toolchain. The build directory should be named "cmake-build-debug-visual-studio" to exempt it from pushing to GitHub. Select "use default" for the generator and "Release" for build type.
- Delete the current cmake-build-debug-visual-studio directory if it exists
- Reload the CMAKE project (file -> reload CMAKE project)
- Build and run the project. Includes should be automatically included in the build directory.

### Visual Studio: 2022 or newer
- Open the project folder
- VS should automatically run CMAKE
- Run the program.
