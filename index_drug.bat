@echo off
setlocal

:: Get Argument or User Input
set DRUG_ID=%~1
if "%DRUG_ID%"=="" (
    set /p DRUG_ID="Enter Drug ID (new folder name in data/raw/): "
)

echo.
echo ================================================================
echo [*] Starting Document Processing and Indexing...
echo     - Drug ID: %DRUG_ID%
echo     - Source: data/raw/%DRUG_ID%/ctd/*.pdf
echo ================================================================
echo.

:: Ensure CTD directory exists
if not exist "data\raw\%DRUG_ID%\ctd" (
    echo [!] ERROR: data\raw\%DRUG_ID%\ctd folder not found!
    echo Please create the folder and place your CTD PDF files there.
    if "%~1"=="" pause
    exit /b 1
)

:: Execute Python Entry Point
call .venv\Scripts\python run_pipeline_step2_5.py --drug %DRUG_ID%

if %errorlevel% neq 0 (
    echo.
    echo [!] ERROR: Indexing failed.
    if "%~1"=="" pause
    exit /b 1
) else (
    echo.
    echo [+] SUCCESS: Indexing complete for %DRUG_ID%.
    echo You can now run tune_section.bat for this drug.
)

if "%~1"=="" pause
