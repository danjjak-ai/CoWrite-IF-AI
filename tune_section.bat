@echo off
setlocal

:: Get User Input
set /p DRUG_ID="Enter Drug ID (folder name in data/raw/): "
set /p SEC_ID="Enter Section ID (e.g., section_VII): "
set /p ITERS="Enter Number of Iterations (e.g., 3): "

echo.
echo ================================================================
echo [*] Starting Section-wise Tuning...
echo     - Drug ID: %DRUG_ID%
echo     - Section: %SEC_ID%
echo     - Iterations: %ITERS%
echo ================================================================
echo.

:: Execute Python Entry Point
.venv\Scripts\python src/tuner/section_tuning_entry.py --drug %DRUG_ID% --section %SEC_ID% --loops %ITERS%

if %errorlevel% neq 0 (
    echo.
    echo [!] ERROR: Tuning process failed. Check your configuration or log.
) else (
    echo.
    echo [+] SUCCESS: Section-wise tuning completed. Results in outputs/%DRUG_ID%/tuning/%SEC_ID%
)

pause
