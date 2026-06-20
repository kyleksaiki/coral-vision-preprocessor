@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM PROJECT FOLDER
REM This uses the folder where run.bat is located.
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

REM Auto-detect OpenCV JAR version, for example opencv-4120.jar
set "OPENCV_JAR="
for %%J in ("!OPENCV_ROOT!\build\java\opencv-*.jar") do (
    if exist "%%~fJ" set "OPENCV_JAR=%%~fJ"
)

REM OpenCV native DLL folder
set "OPENCV_DLL_DIR=!OPENCV_ROOT!\build\java\x64"

REM ============================================================
REM ONNX RUNTIME JAR  (NEW)
REM Drop onnxruntime-1.26.0.jar into a "lib" folder next to run.bat
REM (or leave it in the project root). The jar bundles its own native
REM libraries and unpacks them at runtime, so no extra DLL path is needed.
REM ============================================================
set "ONNXRUNTIME_JAR="
for %%J in ("!PROJECT_DIR!lib\onnxruntime-*.jar" "!PROJECT_DIR!onnxruntime-*.jar") do (
    if exist "%%~fJ" set "ONNXRUNTIME_JAR=%%~fJ"
)

REM ============================================================
REM CARD MODEL PATH  (NEW)
REM Passed to Java via -Dcard.model so the path lives in one place.
REM Defaults to the model in the sibling card_segmenter folder (one level up from
REM the project dir, where run.bat lives); change if you move it. The ..\ is resolved
REM by both Windows and Java.
REM ============================================================
set "CARD_MODEL=!PROJECT_DIR!..\card_segmenter\card_seg.onnx"

REM ============================================================
REM CHECK PROJECT
REM ============================================================
echo.
echo Project folder:
echo   !PROJECT_DIR!
echo.

REM ============================================================
REM CHECK JAVA
REM ============================================================
if not exist "!JAVAC_CMD!" (
    echo.
    echo javac.exe not found:
    echo   !JAVAC_CMD!
    echo.
    echo Check your JDK folder path.
    pause
    exit /b 1
)

if not exist "!JAVA_CMD!" (
    echo.
    echo java.exe not found:
    echo   !JAVA_CMD!
    echo.
    echo Check your JDK folder path.
    pause
    exit /b 1
)

echo Java compiler:
echo   !JAVAC_CMD!
echo.

REM ============================================================
REM CHECK OPENCV
REM ============================================================
if not exist "!OPENCV_ROOT!" (
    echo.
    echo OpenCV folder not found:
    echo   !OPENCV_ROOT!
    echo.
    pause
    exit /b 1
)

if not defined OPENCV_JAR (
    echo.
    echo OpenCV JAR not found in:
    echo   !OPENCV_ROOT!\build\java
    echo.
    echo Expected something like:
    echo   opencv-4120.jar
    echo.
    pause
    exit /b 1
)

if not exist "!OPENCV_DLL_DIR!" (
    echo.
    echo OpenCV DLL folder not found:
    echo   !OPENCV_DLL_DIR!
    echo.
    pause
    exit /b 1
)

echo OpenCV JAR:
echo   !OPENCV_JAR!
echo.
echo OpenCV DLL folder:
echo   !OPENCV_DLL_DIR!
echo.

REM ============================================================
REM CHECK ONNX RUNTIME JAR  (NEW)
REM ============================================================
if not defined ONNXRUNTIME_JAR (
    echo.
    echo ONNX Runtime JAR not found.
    echo Put onnxruntime-1.26.0.jar in:
    echo   !PROJECT_DIR!lib
    echo.
    echo Download it from Maven Central:
    echo   com.microsoft.onnxruntime:onnxruntime:1.26.0
    echo.
    pause
    exit /b 1
)

echo ONNX Runtime JAR:
echo   !ONNXRUNTIME_JAR!
echo.

REM ============================================================
REM CHECK CARD MODEL  (NEW)
REM ============================================================
if not exist "!CARD_MODEL!" (
    echo.
    echo Card model not found:
    echo   !CARD_MODEL!
    echo.
    echo If the file lives in OneDrive and shows a cloud icon, right-click it and choose
    echo   "Always keep on this device"
    echo so Java can actually read it. Otherwise fix CARD_MODEL above.
    echo.
    pause
    exit /b 1
)

echo Card model:
echo   !CARD_MODEL!
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
REM (added CardFinderOnnx.java; added ONNX jar to the classpath)
REM ============================================================
echo Compiling Java files...
echo.

"!JAVAC_CMD!" -d bin -cp ".;!OPENCV_JAR!;!ONNXRUNTIME_JAR!" Main.java TrayCleaner.java TrayLightingNormalizer.java CoralMaskRefiner.java LabelCoral.java LabelAlgae.java LabelSilt.java CardFinder.java CardFinderOnnx.java HoughLines.java

if errorlevel 1 (
    echo.
    echo Compile failed.
    pause
    exit /b 1
)

REM ============================================================
REM ADD OPENCV DLL FOLDER TO PATH
REM (ONNX Runtime needs no DLL path; it unpacks its natives itself)
REM ============================================================
set "PATH=!OPENCV_DLL_DIR!;!PATH!"

REM ============================================================
REM COLLECT INPUT IMAGES
REM ============================================================
set "INPUT_ARGS="

for %%F in (input\*.jpg input\*.jpeg input\*.png input\*.tif input\*.tiff) do (
    if exist "%%F" (
        set INPUT_ARGS=!INPUT_ARGS! "%%~fF"
    )
)

if not defined INPUT_ARGS (
    echo.
    echo No input images found in the input folder.
    echo.
    echo Put input images into:
    echo   !PROJECT_DIR!input
    echo.
    pause
    exit /b 1
)

REM ============================================================
REM RUN PROGRAM
REM (added ONNX jar to -cp; pass the model path via -Dcard.model)
REM ============================================================
echo.
echo Running preprocessor...
echo.

"!JAVA_CMD!" -Djava.library.path="!OPENCV_DLL_DIR!" -cp "bin;!OPENCV_JAR!;!ONNXRUNTIME_JAR!" -Dcard.model="!CARD_MODEL!" Main output !INPUT_ARGS!

if errorlevel 1 (
    echo.
    echo Run failed.
    pause
    exit /b 1
)

echo.
echo Done.
echo Output saved in:
echo   !PROJECT_DIR!output
echo.

pause