@echo off
REM One-time setup: installs a private Python runtime and creates a virtual
REM environment with all dependencies under %LOCALAPPDATA%\FNDetect.
REM Delete that folder to undo. See HOW_TO_RUN.txt.

setlocal EnableExtensions
cd /d "%~dp0"

REM Must stay OUTSIDE the project on a short path: TensorFlow's wheel nests
REM headers ~106 chars deep and hits Windows' 260-char limit otherwise.
set "ENVROOT=%LOCALAPPDATA%\FNDetect"
set "RT=%ENVROOT%\rt"
set "BASEPY=%RT%\python.exe"
set "VENV=%ENVROOT%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "NLTK_DATA=%ENVROOT%\nltk_data"

set "PYURL=https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%%2B20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"

echo.
echo ===================================================================
echo   Fake News Detection System - Setup
echo ===================================================================
echo   Project     : %CD%
echo   Runtime     : %RT%
echo   Environment : %VENV%
echo.

where curl.exe >nul 2>&1
if errorlevel 1 goto NOTOOLS
where tar.exe >nul 2>&1
if errorlevel 1 goto NOTOOLS
goto TOOLSOK

:NOTOOLS
echo [X] This script needs curl.exe and tar.exe, which ship with
echo     Windows 10 build 1803 and newer. Your Windows looks older.
echo     Please update Windows, or install Python 3.12 manually from
echo     https://www.python.org/downloads/ and run:
echo         python -m venv .venv
echo         .venv\Scripts\pip install -r requirements-app.txt
goto FAILED

:TOOLSOK

if exist "%BASEPY%" (
    echo [1/6] Python runtime already present - skipping download.
    goto HAVEPYTHON
)

echo [1/6] Downloading a private Python 3.12 runtime ^(~21 MB^)...
if not exist "%ENVROOT%" mkdir "%ENVROOT%"
curl.exe -L --fail --retry 3 --retry-delay 2 -o "%ENVROOT%\python.tar.gz" "%PYURL%"
if errorlevel 1 (
    echo [X] Download failed. Check your internet connection and retry.
    goto FAILED
)

echo       Extracting...
tar.exe -xzf "%ENVROOT%\python.tar.gz" -C "%ENVROOT%"
if errorlevel 1 (
    echo [X] Could not extract the runtime.
    goto FAILED
)

REM The archive unpacks to a folder called "python"; give it our own name.
if exist "%ENVROOT%\python\python.exe" (
    if exist "%RT%" rmdir /s /q "%RT%"
    move /y "%ENVROOT%\python" "%RT%" >nul
)
del /q "%ENVROOT%\python.tar.gz" >nul 2>&1

if not exist "%BASEPY%" (
    echo [X] The runtime did not unpack as expected.
    goto FAILED
)

:HAVEPYTHON
"%BASEPY%" --version
if errorlevel 1 (
    echo [X] The bundled Python will not start.
    goto FAILED
)

echo.
if exist "%PY%" (
    echo [2/6] Virtual environment already exists - reusing it.
    goto HAVEVENV
)
echo [2/6] Creating the virtual environment...
"%BASEPY%" -m venv "%VENV%"
if errorlevel 1 (
    echo [X] Could not create the virtual environment at:
    echo     %VENV%
    goto FAILED
)

:HAVEVENV
if not exist "%PY%" (
    echo [X] The virtual environment has no python.exe at:
    echo     %PY%
    goto FAILED
)

REM Confirm the venv really is isolated from the base runtime before we
REM install into it, so a broken venv fails here instead of halfway through.
"%PY%" -c "import sys; sys.exit(0 if sys.prefix != sys.base_prefix else 1)"
if errorlevel 1 (
    echo [X] That environment is not isolated from the base runtime.
    echo     Delete %ENVROOT% and run SETUP.bat again.
    goto FAILED
)
echo       Isolated virtual environment ready:
"%PY%" --version

echo.
echo [3/6] Preparing the package installer...
"%PY%" -m ensurepip --upgrade >nul 2>&1
"%PY%" -m pip install --quiet --disable-pip-version-check --no-warn-script-location --upgrade pip setuptools wheel
if errorlevel 1 (
    echo [X] Could not prepare pip inside the virtual environment.
    goto FAILED
)

echo.
echo [4/6] Installing TensorFlow, Streamlit and friends...
echo       ^(about 700 MB - this is the slow part, please wait^)
"%PY%" -m pip install --disable-pip-version-check --no-warn-script-location -r requirements-app.txt
if errorlevel 1 (
    echo.
    echo [X] Dependency installation failed. The most common causes are
    echo     a dropped connection or low disk space. Re-run SETUP.bat -
    echo     it resumes from where it stopped.
    goto FAILED
)

echo.
echo       Installing optional speech-input support...
"%PY%" -m pip install --quiet --disable-pip-version-check --no-warn-script-location SpeechRecognition==3.14.4 PyAudio==0.2.14
if errorlevel 1 (
    echo       [!] Speech input unavailable - the app works fine without it.
)

echo.
echo [5/6] Downloading language data ^(NLTK corpora^)...
if not exist "%NLTK_DATA%" mkdir "%NLTK_DATA%"
"%PY%" -c "from preprocessing import ensure_nltk_data; m = ensure_nltk_data(); print('      missing:', m if m else 'none - all corpora ready')"
if errorlevel 1 (
    echo       [!] Could not pre-download the corpora; the app will retry on first use.
)

echo.
echo [6/6] Verifying the installation...
echo.
"%PY%" verify_setup.py
if errorlevel 1 goto VERIFYFAILED

echo.
echo ===================================================================
echo   SETUP COMPLETE
echo.
echo   Now double-click RUN.bat to start the app.
echo ===================================================================
echo.
pause
exit /b 0

:VERIFYFAILED
echo.
echo ===================================================================
echo   SETUP FINISHED, BUT THE CHECK ABOVE REPORTED PROBLEMS
echo.
echo   Read the [FAIL] lines above - they say exactly what is wrong.
echo.
echo   If TensorFlow could not load its native runtime and the error
echo   mentions an "Application Control policy", Windows Smart App
echo   Control is blocking it - TensorFlow's libraries are unsigned.
echo   Run SETUP.bat once more first; that usually clears it. If it
echo   keeps happening, turn Smart App Control off in:
echo     Windows Security ^> App ^& browser control
echo       ^> Smart App Control settings ^> Off
echo   It cannot be turned back on without reinstalling Windows.
echo.
echo   Without that message, do NOT upgrade TensorFlow - upgrading
echo   causes the other known failure. requirements-app.txt already
echo   pins the working version (2.17.1).
echo ===================================================================
echo.
pause
exit /b 1

:FAILED
echo.
echo ===================================================================
echo   SETUP FAILED - see the message above.
echo ===================================================================
echo.
pause
exit /b 1
