@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ============================================================
REM AI Generated Batch File
REM PROJECT FOLDER
REM This automatically uses the folder where run.bat is located.
REM Users usually do NOT need to change this.
REM
REM If needed, replace this with your own project folder path:
REM set "PROJECT_DIR=C:\Users\yourname\Downloads\your-project-folder"
REM ============================================================
set "PROJECT_DIR=%~dp0"
cd /d "!PROJECT_DIR!"

REM ============================================================
REM OPENCV PATHS
REM Replace these with your own OpenCV paths if they are different.
REM ============================================================
set "OPENCV_JAR=C:\Users\kylek\OneDrive\Desktop\Coral_Vision\opencv\build\java\opencv-4120.jar"
set "OPENCV_DLL_DIR=C:\Users\kylek\OneDrive\Desktop\Coral_Vision\opencv\build\java\x64"

REM Check OpenCV files
if not exist "!OPENCV_JAR!" (
    echo.
    echo OpenCV JAR not found:
    echo   !OPENCV_JAR!
    echo.
    echo Replace OPENCV_JAR in run.bat with your own OpenCV jar path.
    pause
    exit /b 1
)

if not exist "!OPENCV_DLL_DIR!" (
    echo.
    echo OpenCV DLL folder not found:
    echo   !OPENCV_DLL_DIR!
    echo.
    echo Replace OPENCV_DLL_DIR in run.bat with your own OpenCV x64 folder path.
    pause
    exit /b 1
)

REM Create folders if missing
if not exist bin mkdir bin
if not exist input mkdir input
if not exist output mkdir output

REM Clean old tray outputs
rmdir /s /q output\trays 2>nul
mkdir output\trays

REM Clean old compiled files
del /q *.class 2>nul
del /q bin\*.class 2>nul

REM Compile Java files
REM Users should update this line if they rename or add Java files.
javac -d bin -cp ".;!OPENCV_JAR!" Main.java TrayCleaner.java TrayLightingNormalizer.java LabelCoral.java LabelAlgae.java LabelSilt.java

if errorlevel 1 (
    echo.
    echo Compile failed.
    pause
    exit /b 1
)

REM Add OpenCV native DLL folder to PATH
set "PATH=!OPENCV_DLL_DIR!;!PATH!"

set "INPUT_ARGS="

REM Collect supported input image files
for %%F in (input\*.jpg input\*.jpeg input\*.png input\*.tif input\*.tiff) do (
    if exist "%%F" (
        set "INPUT_ARGS=!INPUT_ARGS! "%%F""
    )
)

if not defined INPUT_ARGS (
    echo.
    echo No input images found in the input folder.
    echo Put input images into:
    echo   !PROJECT_DIR!input
    echo.
    pause
    exit /b 1
)

REM Run the preprocessor
java -cp "bin;!OPENCV_JAR!" Main output !INPUT_ARGS!

if errorlevel 1 (
    echo.
    echo Run failed.
    pause
    exit /b 1
)

echo.
echo Done.
pause
