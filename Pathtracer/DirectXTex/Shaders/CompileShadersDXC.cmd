@echo off
rem Copyright (c) Microsoft Corporation.
rem Licensed under the MIT License.
rem FINAL KORRIGIERTE VERSION, UM EINE LOKALE KOPIE VON FXC.EXE ZU VERWENDEN

setlocal
set error=0

REM Setze den Pfad zu unserer lokalen Kopie von fxc.exe.
set PCFXC=.\fxc.exe

REM Prüfe, ob die exe-Datei existiert.
if not exist "%PCFXC%" (
    echo.
    echo FEHLER: Konnte fxc.exe nicht unter dem Pfad finden: %PCFXC%
    echo Bitte stelle sicher, dass fxc.exe und d3dcompiler_47.dll im 'tools'-Ordner deines Projekts liegen.
    echo.
    pause
    exit /b 1
)

echo Verwende lokalen Compiler: %PCFXC%
echo.

set FXCOPTS=/nologo /WX /Ges /Zi /Zpc /Qstrip_reflect /Qstrip_debug

call :CompileShader BC7Encode TryMode456CS
call :CompileShader BC7Encode TryMode137CS
call :CompileShader BC7Encode TryMode02CS
call :CompileShader BC7Encode EncodeBlockCS

call :CompileShader BC6HEncode TryModeG10CS
call :CompileShader BC6HEncode TryModeLE10CS
call :CompileShader BC6HEncode EncodeBlockCS

echo.

if %error% == 0 (
    echo Alle Shader erfolgreich kompiliert!
    echo Die .inc-Dateien befinden sich jetzt im 'DirectXTex'-Ordner.
) else (
    echo Es gab Fehler bei der Shader-Kompilierung!
)

echo.
pause
endlocal
exit /b 0

:CompileShader
REM Wir schreiben die .inc-Dateien in das übergeordnete Verzeichnis.
set fxc=%PCFXC% "%1.hlsl" %FXCOPTS% /Tcs_5_0 /E%2 "/Fh..\%1_%2.inc" "/Fd..\PDB\%1_%2.pdb" /Vn%1_%2

REM KORREKTUR: Der Variablenname (-Vn) darf NICHT das Suffix "_cs40" haben. Nur der Dateiname.
set fxc4=%PCFXC% "%1.hlsl" %FXCOPTS% /Tcs_4_0 /DEMULATE_F16C /E%2 "/Fh..\%1_%2_cs40.inc" "/Fd..\PDB\%1_%2_cs40.pdb" /Vn%1_%2

echo %fxc%
%fxc% || set error=1
echo %fxc4%
%fxc4% || set error=1
exit /b