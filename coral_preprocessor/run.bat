@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM PROJECT FOLDER
REM ============================================================
set "PROJECT_DIR=%~dp0"
cd /d "!PROJECT_DIR!"

REM ============================================================
REM JAVA / JDK PATH
REM ============================================================
set "JDK_DIR=C:\Program Files\Eclipse Adoptium\jdk-21.0.11.10-hotspot"
set "JAVAC_CMD=!JDK_DIR!\bin\javac.exe"
set "JAVA_CMD=!JDK_DIR!\bin\java.exe"

REM ============================================================
REM OPENCV PATHS
REM ============================================================
set "OPENCV_ROOT=C:\Users\kylek\OneDrive\Desktop\Personal_Projects\opencv"

set "OPENCV_JAR="
for %%J in ("!OPENCV_ROOT!\build\java\opencv-*.jar") do (
    if exist "%%~fJ" set "OPENCV_JAR=%%~fJ"
)
set "OPENCV_DLL_DIR=!OPENCV_ROOT!\build\java\x64"

REM ============================================================
REM ONNX RUNTIME JAR
REM Drop onnxruntime-1.26.0.jar into a "lib" folder next to run.bat.
REM ============================================================
set "ONNXRUNTIME_JAR="
for %%J in ("!PROJECT_DIR!lib\onnxruntime-*.jar" "!PROJECT_DIR!onnxruntime-*.jar") do (
    if exist "%%~fJ" set "ONNXRUNTIME_JAR=%%~fJ"
)

REM ============================================================
REM MODEL PATHS  (passed to Java via -Dcard.model / -Dcoral.model)
REM Default to sibling folders one level up from the project dir.
REM ============================================================
set "CARD_MODEL=!PROJECT_DIR!..\card_segmenter\card_seg.onnx"
set "CORAL_MODEL=!PROJECT_DIR!..\coral_segmenter\coral_seg.onnx"

echo.
echo Project folder:
echo   !PROJECT_DIR!
echo.

REM ============================================================
REM CHECK JAVA
REM ============================================================
if not exist "!JAVAC_CMD!" ( echo javac.exe not found: !JAVAC_CMD! & pause & exit /b 1 )
if not exist "!JAVA_CMD!"  ( echo java.exe not found: !JAVA_CMD!   & pause & exit /b 1 )
echo Java compiler:
echo   !JAVAC_CMD!
echo.

REM ============================================================
REM CHECK OPENCV
REM ============================================================
if not exist "!OPENCV_ROOT!" ( echo OpenCV folder not found: !OPENCV_ROOT! & pause & exit /b 1 )
if not defined OPENCV_JAR    ( echo OpenCV JAR not found in: !OPENCV_ROOT!\build\java & pause & exit /b 1 )
if not exist "!OPENCV_DLL_DIR!" ( echo OpenCV DLL folder not found: !OPENCV_DLL_DIR! & pause & exit /b 1 )
echo OpenCV JAR:
echo   !OPENCV_JAR!
echo OpenCV DLL folder:
echo   !OPENCV_DLL_DIR!
echo.

REM ============================================================
REM CHECK ONNX RUNTIME JAR
REM ============================================================
if not defined ONNXRUNTIME_JAR (
    echo.
    echo ONNX Runtime JAR not found.
    echo Put onnxruntime-1.26.0.jar in:
    echo   !PROJECT_DIR!lib
    echo Download: com.microsoft.onnxruntime:onnxruntime:1.26.0
    echo.
    pause
    exit /b 1
)
echo ONNX Runtime JAR:
echo   !ONNXRUNTIME_JAR!
echo.

REM ============================================================
REM CHECK MODELS
REM ============================================================
if not exist "!CARD_MODEL!" (
    echo.
    echo Card model not found:
    echo   !CARD_MODEL!
    echo If it lives in OneDrive with a cloud icon, right-click -^> "Always keep on this device".
    echo.
    pause
    exit /b 1
)
if not exist "!CORAL_MODEL!" (
    echo.
    echo Coral model not found:
    echo   !CORAL_MODEL!
    echo If it lives in OneDrive with a cloud icon, right-click -^> "Always keep on this device".
    echo.
    pause
    exit /b 1
)
echo Card model:
echo   !CARD_MODEL!
echo Coral model:
echo   !CORAL_MODEL!
echo.

REM ============================================================
REM CREATE NEEDED FOLDERS
REM ============================================================
if not exist bin mkdir bin
if not exist input mkdir input
if not exist output mkdir output

REM ============================================================
REM CLEAN OLD OUTPUTS
REM ============================================================
rmdir /s /q output\trays 2>nul
mkdir output\trays
del /q *.class 2>nul
del /q bin\*.class 2>nul

REM ============================================================
REM COMPILE JAVA FILES
REM (classical labelers removed; CardFinderOnnx + CoralFinderOnnx added)
REM ============================================================
echo Compiling Java files...
echo.
"!JAVAC_CMD!" -d bin -cp ".;!OPENCV_JAR!;!ONNXRUNTIME_JAR!" Main.java TrayCleaner.java TrayLightingNormalizer.java CardFinder.java CardFinderOnnx.java CoralFinderOnnx.java

if errorlevel 1 ( echo. & echo Compile failed. & pause & exit /b 1 )

REM ============================================================
REM ADD OPENCV DLL FOLDER TO PATH (ONNX unpacks its own natives)
REM ============================================================
set "PATH=!OPENCV_DLL_DIR!;!PATH!"

REM ============================================================
REM COLLECT INPUT IMAGES
REM ============================================================
set "INPUT_ARGS="
for %%F in (input\*.jpg input\*.jpeg input\*.png input\*.tif input\*.tiff) do (
    if exist "%%F" set INPUT_ARGS=!INPUT_ARGS! "%%~fF"
)
if not defined INPUT_ARGS (
    echo.
    echo No input images found in: !PROJECT_DIR!input
    echo.
    pause
    exit /b 1
)

REM ============================================================
REM RUN PROGRAM
REM ============================================================
echo.
echo Running preprocessor...
echo.
"!JAVA_CMD!" -Djava.library.path="!OPENCV_DLL_DIR!" -cp "bin;!OPENCV_JAR!;!ONNXRUNTIME_JAR!" -Dcard.model="!CARD_MODEL!" -Dcoral.model="!CORAL_MODEL!" Main output !INPUT_ARGS!

if errorlevel 1 ( echo. & echo Run failed. & pause & exit /b 1 )

echo.
echo Done.
echo Output saved in:
echo   !PROJECT_DIR!output
echo.
pause