@echo off
REM One-time setup: creates a virtual environment with all dependencies under
REM %LOCALAPPDATA%\FNDetect. Delete that folder to undo. See HOW_TO_RUN.txt.

setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

REM Must stay OUTSIDE the project on a short path: TensorFlow's wheel nests
REM headers ~106 chars deep and hits Windows' 260-char limit otherwise.
set "ENVROOT=%LOCALAPPDATA%\FNDetect"
set "RT=%ENVROOT%\rt"
set "VENV=%ENVROOT%\venv"
set "PY=%VENV%\Scripts\python.exe"
set "NLTK_DATA=%ENVROOT%\nltk_data"
set "PROBE=%TEMP%\fnd_pyprobe.txt"

set "PYURL=https://github.com/astral-sh/python-build-standalone/releases/download/20260901/cpython-3.12.14%%2B20260901-x86_64-pc-windows-msvc-install_only_stripped.tar.gz"

echo.
echo ===================================================================
echo   Fake News Detection System - Setup
echo ===================================================================
echo   Project     : %CD%
echo   Environment : %VENV%
echo.

echo [1/6] Looking for a usable Python 3.10-3.12...

REM An official python.org build is strongly preferred. Its binaries are signed
REM by the Python Software Foundation, so Windows Smart App Control trusts them.
REM The downloadable standalone runtime is unsigned, and Smart App Control
REM blocks its extension modules (_multiprocessing and friends) on some PCs.
set "BASEPY="
call :FINDPY py -3.12
if not defined BASEPY call :FINDPY py -3.11
if not defined BASEPY call :FINDPY py -3.10
if not defined BASEPY call :FINDPY python
if not defined BASEPY call :FINDPY "%RT%\python.exe"

if defined BASEPY (
    echo       Using: !BASEPY!
    goto HAVEPYTHON
)

echo       No suitable Python found - downloading a private one ^(~21 MB^).
echo       [!] It is unsigned, so if Windows Smart App Control is on it may
echo           block it. Installing Python 3.12 from python.org and re-running
echo           SETUP.bat avoids that entirely.

where curl.exe >nul 2>&1
if errorlevel 1 goto NOTOOLS
where tar.exe >nul 2>&1
if errorlevel 1 goto NOTOOLS

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

call :FINDPY "%RT%\python.exe"
if not defined BASEPY (
    echo [X] The downloaded runtime did not unpack as expected.
    goto FAILED
)
echo       Using: !BASEPY!

:HAVEPYTHON
echo.

REM Reuse an existing environment only if it was built on the Python we just
REM chose. One left over from the unsigned runtime has to be rebuilt, and
REM comparing the base prefix is reliable where probing for a Smart App Control
REM block is not - those blocks come and go between runs.
if exist "%PY%" (
    set "VBASE="
    set "NBASE="
    "%PY%" -c "import sys;sys.stdout.write(sys.base_prefix)" > "%PROBE%" 2>nul
    set /p VBASE=<"%PROBE%"
    "%BASEPY%" -c "import sys;sys.stdout.write(sys.prefix)" > "%PROBE%" 2>nul
    set /p NBASE=<"%PROBE%"
    if /i "!VBASE!"=="!NBASE!" (
        echo [2/6] Virtual environment already exists - reusing it.
        goto HAVEVENV
    )
    echo [2/6] Existing environment was built on a different Python - rebuilding.
    rmdir /s /q "%VENV%" 2>nul
    if exist "%VENV%" (
        echo [X] The old environment could not be removed. It is usually still
        echo     in use - close RUN.bat and any window running the app, then
        echo     run SETUP.bat again. If that does not help, delete this by
        echo     hand and re-run:
        echo       %VENV%
        goto FAILED
    )
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

REM Fail here, not halfway through a 700 MB install, if anything is wrong.
REM Written without "!=" on purpose: delayed expansion eats the "!".
"%PY%" -c "import sys; sys.exit(sys.prefix == sys.base_prefix)"
if errorlevel 1 (
    echo [X] That environment is not isolated from its base Python.
    echo     Delete %ENVROOT% and run SETUP.bat again.
    goto FAILED
)
"%PY%" -c "import _multiprocessing, ssl, ctypes" 2>nul
if errorlevel 1 (
    echo [X] The environment cannot load core Python modules.
    echo     This is usually Windows Smart App Control blocking an unsigned
    echo     Python. Install Python 3.12 from https://www.python.org/downloads/
    echo     then delete %ENVROOT% and run SETUP.bat again.
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


REM --- helper: accept a Python only if it is 3.10-3.12 and can run --------
:FINDPY
if defined BASEPY goto :eof
set "CAND="
%* -c "import sys;sys.stdout.write(sys.executable if (3,10)<=sys.version_info[:2]<(3,13) else '')" > "%PROBE%" 2>nul
if errorlevel 1 goto :eof
for %%S in ("%PROBE%") do if %%~zS EQU 0 goto :eof
set /p CAND=<"%PROBE%"
if not defined CAND goto :eof
set "BASEPY=%CAND%"
goto :eof


:NOTOOLS
echo [X] No Python was found, and this script needs curl.exe and tar.exe
echo     to fetch one. Those ship with Windows 10 build 1803 and newer.
echo     Please install Python 3.12 from https://www.python.org/downloads/
echo     and run SETUP.bat again.
goto FAILED

:VERIFYFAILED
echo.
echo ===================================================================
echo   SETUP FINISHED, BUT THE CHECK ABOVE REPORTED PROBLEMS
echo.
echo   Read the [FAIL] lines above - they say exactly what is wrong.
echo.
echo   If the error mentions an "Application Control policy", Windows
echo   Smart App Control is blocking an unsigned file. The usual fix is
echo   to install Python 3.12 from https://www.python.org/downloads/
echo   (its binaries are signed), then delete
echo     %ENVROOT%
echo   and run SETUP.bat again so the environment is rebuilt on it.
echo.
echo   Do NOT upgrade TensorFlow to fix a load error - upgrading causes
echo   the other known failure. requirements-app.txt pins 2.17.1.
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
