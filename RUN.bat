@echo off
REM Starts the app at http://localhost:8501. Runs SETUP.bat first if the
REM environment is missing. Close this window to stop. See HOW_TO_RUN.txt.

setlocal EnableExtensions
cd /d "%~dp0"

set "ENVROOT=%LOCALAPPDATA%\FNDetect"
set "VENV=%ENVROOT%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "NLTK_DATA=%ENVROOT%\nltk_data"

REM Quieten TensorFlow startup chatter so real errors stand out.
set "TF_CPP_MIN_LOG_LEVEL=3"
set "TF_ENABLE_ONEDNN_OPTS=0"
set "PYTHONIOENCODING=utf-8"

if not exist "%PY%" (
    echo.
    echo  The app is not set up on this computer yet.
    echo  Running SETUP.bat first - this takes 5-10 minutes and needs internet.
    echo.
    pause
    call "%~dp0SETUP.bat"
    if not exist "%PY%" (
        echo.
        echo  Setup did not complete, so the app cannot start.
        echo.
        pause
        exit /b 1
    )
)

if not exist "%~dp0app.py" (
    echo.
    echo  [X] app.py is missing. Keep RUN.bat inside the project folder,
    echo      next to app.py.
    echo.
    pause
    exit /b 1
)

echo.
echo ===================================================================
echo   Fake News Detection System
echo ===================================================================
echo   Starting... your browser will open at http://localhost:8501
echo   Leave this window open while you use the app.
echo   Press Ctrl+C or close this window to stop.
echo ===================================================================
echo.

"%PY%" -m streamlit run app.py
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo ===================================================================
    echo   The app stopped with an error ^(code %RC%^).
    echo.
    echo   Run this to find out why:
    echo       SETUP.bat
    echo   It re-checks every dependency and prints a diagnosis.
    echo ===================================================================
    echo.
    pause
)

exit /b %RC%
