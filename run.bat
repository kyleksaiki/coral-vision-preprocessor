@echo off
setlocal EnableDelayedExpansion

cd /d "C:\Users\kylek\OneDrive\Desktop\Coral_Vision\preprocessor"

if not exist bin mkdir bin
if not exist input mkdir input
if not exist output mkdir output

rmdir /s /q output\cards 2>nul
mkdir output\cards

del /q *.class 2>nul
del /q bin\*.class 2>nul

javac -d bin -cp ".;C:\Users\kylek\OneDrive\Desktop\Coral_Vision\opencv\build\java\opencv-4120.jar" Main.java CardCleaner.java CardLightingNormalizer.java LabelCoral.java LabelAlgae.java LabelSilt.java 

if errorlevel 1 (
    echo.
    echo Compile failed.
    pause
    exit /b 1
)

set "PATH=C:\Users\kylek\OneDrive\Desktop\Coral_Vision\opencv\build\java\x64;%PATH%"

set "INPUT_ARGS="

for %%F in (input\*.jpg input\*.jpeg input\*.png input\*.tif input\*.tiff) do (
    if exist "%%F" (
        set "INPUT_ARGS=!INPUT_ARGS! "%%F""
    )
)

if not defined INPUT_ARGS (
    echo.
    echo No input images found in the input folder.
    echo Put input images into:
    echo   C:\Users\kylek\OneDrive\Desktop\Coral_Vision\preprocessor\input
    echo.
    pause
    exit /b 1
)

java -cp "bin;C:\Users\kylek\OneDrive\Desktop\Coral_Vision\opencv\build\java\opencv-4120.jar" Main output !INPUT_ARGS!

if errorlevel 1 (
    echo.
    echo Run failed.
    pause
    exit /b 1
)

echo.
echo Done.
pause
