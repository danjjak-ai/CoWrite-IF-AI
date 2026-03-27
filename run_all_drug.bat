@echo off
setlocal enabledelayedexpansion

set DRUG_ID=%~1
set REF_PDF=%~2

if "%DRUG_ID%"=="" (
    echo [ERROR] Missing arguments.
    echo Usage: run_all_drug.bat ^<drug_id^> ^<reference_pdf_name^>
    echo Example: run_all_drug.bat drug_B IF00009405.pdf
    exit /b 1
)

if "%REF_PDF%"=="" (
    echo [ERROR] Missing reference PDF parameter.
    echo Usage: run_all_drug.bat ^<drug_id^> ^<reference_pdf_name^>
    echo Example: run_all_drug.bat drug_B IF00009405.pdf
    exit /b 1
)

echo.
echo ====================================================================
echo [STEP 0] Initialization
echo Target Drug ID: %DRUG_ID%
echo Target Ref PDF: data\raw\%DRUG_ID%\%REF_PDF%
echo ====================================================================
echo.

echo [STEP 1] Parsing the Human Reference IF PDF...
echo.
call .venv\Scripts\python parse_reference_if.py --target_dir data/raw/%DRUG_ID% --reference_file %REF_PDF%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Step 1 Failed.
    exit /b 1
)
echo.

echo [STEP 2] Indexing the CTD Original Documents...
echo.
call index_drug.bat %DRUG_ID%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Step 2 Failed.
    exit /b 1
)
echo.

echo [STEP 3] Running the Baseline Generation Pipeline...
echo.
call .venv\Scripts\python main.py --mode tune --drug_id %DRUG_ID%
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Step 3 Failed.
    exit /b 1
)

echo.
echo ====================================================================
echo SUCCESS: Pipeline completed for %DRUG_ID%!
echo Please check the dashboard.
echo ====================================================================
exit /b 0
