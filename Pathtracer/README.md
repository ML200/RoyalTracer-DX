### Setup Instructions:
1. Clone this repository
2. Install the windows sdk https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
3. Make sure to have the visual studio cpp compiler installed
4. In CLion go to settings -> Build, Execution, Deployment -> Toolchains and select the Visual Studio compiler
5. In CLion go to settings -> Build, Execution, Deployment -> CMake and select visual studio as toolchain
6. Make sure to compile the project as a 64 bit application. In the toolchain settings for the visual studio compiler, you can set the architecture
7. Build the project
8. Run the project