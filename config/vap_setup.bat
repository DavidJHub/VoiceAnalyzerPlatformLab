@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: VAP - Environment Setup v2.3.0
:: Location : <PROJECT_ROOT>\config\vap_setup.bat
:: Venv      : <PROJECT_ROOT>\.vap_env
:: ============================================================
set "SCRIPT_VERSION=2.3.0"

:: --- Path resolution -----------------------------------------
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%i in ("%SCRIPT_DIR%") do set "PROJECT_ROOT=%%~dpi"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

:: Venv lives OUTSIDE the repo: one level above PROJECT_ROOT
for %%i in ("%PROJECT_ROOT%") do set "REPO_PARENT=%%~dpi"
set "REPO_PARENT=%REPO_PARENT:~0,-1%"
set "ENV_DIR=%REPO_PARENT%\.vap_env"
set "LOG_DIR=%PROJECT_ROOT%\logs"
set "FFMPEG_DIR=%PROJECT_ROOT%\ffmpeg"
set "TEMP_DIR=%PROJECT_ROOT%\_temp_ffmpeg"
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

:: --- Locale-safe timestamp via PowerShell --------------------
:: %date% format varies by Windows locale (es-ES returns "do 08/03/2026")
:: PowerShell always returns ISO format regardless of locale
for /f "delims=" %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%t"

:: --- Create dirs BEFORE first log write ----------------------
if not exist "%LOG_DIR%"              mkdir "%LOG_DIR%"
if not exist "%PROJECT_ROOT%\process" mkdir "%PROJECT_ROOT%\process"
if not exist "%PROJECT_ROOT%\models"  mkdir "%PROJECT_ROOT%\models"

set "LOG_FILE=%LOG_DIR%\setup_%TIMESTAMP%.log"
echo [%TIMESTAMP%] === VAP Setup v%SCRIPT_VERSION% === > "%LOG_FILE%"

:: ============================================================
:: ARGS
:: ============================================================
set "HF_TOKEN="
set "VOSK_SIZE=small"
set "SKIP_PYANNOTE="
set "SKIP_VOSK="

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--hf-token"      ( set "HF_TOKEN=%~2"    & shift & shift & goto :parse_args )
if /i "%~1"=="--vosk-size"     ( set "VOSK_SIZE=%~2"   & shift & shift & goto :parse_args )
if /i "%~1"=="--skip-pyannote" ( set "SKIP_PYANNOTE=1" & shift & goto :parse_args )
if /i "%~1"=="--skip-vosk"     ( set "SKIP_VOSK=1"     & shift & goto :parse_args )
shift & goto :parse_args
:args_done

if "%HF_TOKEN%"=="" (
    if exist "%PROJECT_ROOT%\.env" (
        for /f "tokens=1,2 delims==" %%a in ('findstr /i "HF_TOKEN" "%PROJECT_ROOT%\.env"') do set "HF_TOKEN=%%b"
    )
)

call :log "INFO" "Script dir   : %SCRIPT_DIR%"
call :log "INFO" "Project root : %PROJECT_ROOT%"
call :log "INFO" "Repo parent  : %REPO_PARENT%"
call :log "INFO" "Venv dir     : %ENV_DIR%"
call :log "INFO" "VOSK_SIZE    : %VOSK_SIZE%"

echo.
echo  +----------------------------------------------+
echo  ^|       VAP Environment Setup v%SCRIPT_VERSION%        ^|
echo  +----------------------------------------------+
echo  Project root : %PROJECT_ROOT%
echo  Venv         : %ENV_DIR%
echo  Log          : %LOG_FILE%
echo.

:: ============================================================
:: STEP 1 - Detect Python 3.11
:: IMPORTANT: Do NOT call "py -X.XX" directly — on Store Python
:: it opens the Microsoft Store and hangs indefinitely.
:: Strategy: only call py.exe if "where py" confirms it exists,
:: then verify version via python -c (no stdout pipe issues).
:: ============================================================
call :header "STEP 1/5" "Detecting Python 3.11"

set "PYTHON_EXE="
set "PY_VER="

:: --- Candidate 1: py launcher (python.org install only) ---
where py >nul 2>&1
if not errorlevel 1 (
    call :log "INFO" "py.exe found — checking version"
    :: Use python -c to get version, avoids pipe+findstr errorlevel issues
    py -3.11 -c "import sys; v=sys.version_info; sys.exit(0 if v.major==3 and v.minor==11 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py -3.11"
        for /f "tokens=*" %%v in ('py -3.11 -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
        goto :python_found
    )
    call :log "WARN" "py.exe found but not 3.11 — trying next"
)

:: --- Candidate 2: python in PATH ---
where python >nul 2>&1
if not errorlevel 1 (
    call :log "INFO" "python found — checking version"
    python -c "import sys; v=sys.version_info; sys.exit(0 if v.major==3 and v.minor==11 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
        for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
        goto :python_found
    )
    for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "_FOUND_VER=%%v"
    call :log "WARN" "python found but version is !_FOUND_VER! (need 3.11) — trying next"
)

:: --- Candidate 3: python3.11 explicit ---
where python3.11 >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python3.11"
    for /f "tokens=*" %%v in ('python3.11 -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
    goto :python_found
)

call :log "ERROR" "Python 3.11 not found via py, python, or python3.11"
echo.
echo  [FATAL] Python 3.11 not found.
echo          - Official install : https://python.org/downloads/
echo          - Microsoft Store  : search "Python 3.11" in the Store
echo          - Ensure Python is checked in PATH during install
echo.
pause & exit /b 1

:python_found
call :log "OK" "Python %PY_VER% via [%PYTHON_EXE%]"

:: ============================================================
:: STEP 2 - Virtual environment
:: ============================================================
call :header "STEP 2/5" "Virtual environment"

:: Full integrity check: python.exe + activate.bat + executes cleanly
set "VENV_OK=0"
if exist "%ENV_DIR%\Scripts\python.exe" (
    if exist "%ENV_DIR%\Scripts\activate.bat" (
        "%ENV_DIR%\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>&1
        if not errorlevel 1 set "VENV_OK=1"
    )
)

if "!VENV_OK!"=="0" (
    if exist "%ENV_DIR%" (
        call :log "WARN" "Incomplete or broken venv at %ENV_DIR% — removing"
        rmdir /s /q "%ENV_DIR%" >> "%LOG_FILE%" 2>&1
        if exist "%ENV_DIR%" (
            call :log "ERROR" "Cannot remove %ENV_DIR%"
            echo  [FATAL] Cannot delete broken venv. Close any terminals/processes using it.
            pause & exit /b 1
        )
        call :log "OK" "Broken venv removed"
    )
    call :log "INFO" "Creating venv at %ENV_DIR%"
    %PYTHON_EXE% -m venv "%ENV_DIR%" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log "ERROR" "venv creation failed"
        pause & exit /b 1
    )
    call :log "OK" "Venv created"
) else (
    call :log "SKIP" "Venv OK — reusing"
)

:: Activate
call "%ENV_DIR%\Scripts\activate.bat"

:: Verify via the venv python directly — do NOT rely on PATH resolution after activation
set "VERIFY_PY=%ENV_DIR%\Scripts\python.exe"
"%VERIFY_PY%" -c "import sys; sys.exit(0)" >nul 2>&1
if errorlevel 1 (
    call :log "ERROR" "Venv python.exe is not functional after activation"
    pause & exit /b 1
)

for /f "delims=" %%p in ('"%VERIFY_PY%" -c "import sys; print(sys.executable)"') do set "ACTIVE_PY=%%p"
call :log "OK" "Venv active: !ACTIVE_PY!"

:: From here, always call venv python directly to avoid PATH shadowing
set "VENV_PY=%ENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%ENV_DIR%\Scripts\pip.exe"

:: ============================================================
:: STEP 3 - Dependencies
:: ============================================================
call :header "STEP 3/5" "Python dependencies"

call :log "INFO" "Upgrading pip/setuptools/wheel"
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "ERROR" "pip upgrade failed. See %LOG_FILE%"
    pause & exit /b 1
)

:: Find requirements.txt (config\ first, then root)
set "REQ_FILE=%SCRIPT_DIR%\requirements.txt"
if not exist "%REQ_FILE%" set "REQ_FILE=%PROJECT_ROOT%\requirements.txt"
if not exist "%REQ_FILE%" (
    call :log "ERROR" "requirements.txt not found in %SCRIPT_DIR% or %PROJECT_ROOT%"
    pause & exit /b 1
)
call :log "INFO" "requirements.txt : %REQ_FILE%"

call :log "INFO" "Installing webrtcvad-wheels (pre-built, avoids MSVC)"
"%VENV_PIP%" install webrtcvad-wheels >> "%LOG_FILE%" 2>&1
if errorlevel 1 call :log "WARN" "webrtcvad-wheels failed — source build will be attempted"

call :log "INFO" "pip install -r requirements.txt"
"%VENV_PIP%" install -r "%REQ_FILE%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "ERROR" "pip install failed. Check %LOG_FILE%"
    pause & exit /b 1
)
call :log "OK" "Dependencies installed"

:: ============================================================
:: STEP 4 - ffmpeg
:: ============================================================
call :header "STEP 4/5" "ffmpeg"

where ffmpeg >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%p in ('where ffmpeg') do call :log "SKIP" "Already on PATH: %%p"
    goto :ffmpeg_done
)
if exist "%FFMPEG_DIR%\ffmpeg.exe" goto :ffmpeg_set_path

if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"
call :log "INFO" "Downloading ffmpeg"
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%TEMP_DIR%\ffmpeg.zip' -UseBasicParsing" >> "%LOG_FILE%" 2>&1
if errorlevel 1 ( call :log "ERROR" "ffmpeg download failed" & pause & exit /b 1 )

powershell -NoProfile -Command "Expand-Archive -Path '%TEMP_DIR%\ffmpeg.zip' -DestinationPath '%TEMP_DIR%' -Force" >> "%LOG_FILE%" 2>&1
if errorlevel 1 ( call :log "ERROR" "ffmpeg extract failed" & pause & exit /b 1 )

if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"
for /D %%f in ("%TEMP_DIR%\ffmpeg-*") do (
    xcopy /E /I /Y "%%f\bin\*" "%FFMPEG_DIR%\" >> "%LOG_FILE%" 2>&1
    goto :ffmpeg_copied
)
:ffmpeg_copied
rmdir /s /q "%TEMP_DIR%"
call :log "OK" "Extracted to %FFMPEG_DIR%"

:ffmpeg_set_path
for /f "skip=2 tokens=3*" %%a in ('reg query HKCU\Environment /v PATH 2^>nul') do set "_UP=%%a %%b"
echo !_UP! | find /i "%FFMPEG_DIR%" >nul 2>&1
if errorlevel 1 (
    setx PATH "!_UP!;%FFMPEG_DIR%" >> "%LOG_FILE%" 2>&1
    call :log "OK" "ffmpeg added to user PATH permanently"
) else (
    call :log "SKIP" "Already in user PATH"
)
set "PATH=%FFMPEG_DIR%;%PATH%"

:ffmpeg_done
call :log "OK" "ffmpeg ready"

:: ============================================================
:: STEP 5 - Models
:: ============================================================
call :header "STEP 5/5" "Models and language data"

if not exist "%PROJECT_ROOT%\.env" ( echo # VAP config > "%PROJECT_ROOT%\.env" )

if not "%HF_TOKEN%"=="" (
    findstr /i "HF_TOKEN" "%PROJECT_ROOT%\.env" >nul 2>&1
    if errorlevel 1 (
        echo HF_TOKEN=%HF_TOKEN% >> "%PROJECT_ROOT%\.env"
        call :log "OK" "HF_TOKEN written to .env"
    ) else (
        call :log "SKIP" "HF_TOKEN already in .env"
    )
) else (
    call :log "WARN" "No --hf-token. pyannote models will be skipped."
    call :log "WARN" "Accept terms : https://hf.co/pyannote/speaker-diarization-3.1"
    call :log "WARN" "Re-run with  : vap_setup.bat --hf-token hf_xxxxxxxxxx"
)

set "MODELS_SCRIPT=%SCRIPT_DIR%\_setup_models.py"
if not exist "%MODELS_SCRIPT%" (
    call :log "ERROR" "_setup_models.py not found at %MODELS_SCRIPT%"
    pause & exit /b 1
)

set "MODEL_ARGS=--vosk-size %VOSK_SIZE%"
if not "%HF_TOKEN%"==""   set "MODEL_ARGS=%MODEL_ARGS% --hf-token %HF_TOKEN%"
if "%SKIP_PYANNOTE%"=="1" set "MODEL_ARGS=%MODEL_ARGS% --skip-pyannote"
if "%SKIP_VOSK%"=="1"     set "MODEL_ARGS=%MODEL_ARGS% --skip-vosk"

call :log "INFO" "Running _setup_models.py %MODEL_ARGS%"
"%VENV_PY%" "%MODELS_SCRIPT%" %MODEL_ARGS%
if errorlevel 1 call :log "WARN" "_setup_models.py had errors — check console output above"

:: ============================================================
:: DONE
:: ============================================================
echo.
echo  +----------------------------------------------+
echo  ^|              Setup complete!                 ^|
echo  +----------------------------------------------+
echo.
echo   Root   : %PROJECT_ROOT%
echo   Venv   : %ENV_DIR%
echo   Log    : %LOG_FILE%
echo   .env   : %PROJECT_ROOT%\.env
echo   Models : %PROJECT_ROOT%\models\
echo.
echo   Activate in a new terminal:
echo     call "%ENV_DIR%\Scripts\activate.bat"
echo.
if "%HF_TOKEN%"=="" (
    echo  [!] pyannote NOT downloaded - no HF token.
    echo      Re-run: vap_setup.bat --hf-token hf_xxxxxxxxxx
    echo.
)
call :log "INFO" "=== Finished ==="
pause
exit /b 0

:: ============================================================
:: SUBROUTINES
:: ============================================================
:header
echo.
echo  -- %~1 : %~2
call :log "INFO" "[%~1] %~2"
exit /b 0

:log
echo  [%~1] %~2
echo [%TIMESTAMP%] [%~1] %~2 >> "%LOG_FILE%"
exit /b 0