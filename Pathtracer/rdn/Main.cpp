//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

#include <iostream>

#include "stdafx.h"
#include "Renderer.h"

_Use_decl_annotations_
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE, LPSTR, int nCmdShow)
{
    if (AllocConsole()) {
        freopen("CONOUT$", "w", stdout);
        freopen("CONOUT$", "w", stderr);
        std::wcout << L"Console initialized" << std::endl;
    }

    sl::Preferences pref;
    pref.showConsole = true;                        // for debugging, set to false in production
    pref.logLevel = sl::LogLevel();
    pref.pathsToPlugins = {}; // change this if Streamline plugins are not located next to the executable
    pref.numPathsToPlugins = 0; // change this if Streamline plugins are not located next to the executable
    pref.pathToLogsAndData = {};                    // change this to enable logging to a file
    slInit(pref,sl::kSDKVersion);

	Renderer sample(1920, 1080, L"DXR Pathtracer - experimental");
	return Win32Application::Run(&sample, hInstance, nCmdShow);
}
