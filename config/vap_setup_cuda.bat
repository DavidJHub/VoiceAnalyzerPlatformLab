@echo off
setlocal EnableDelayedExpansion

:: ============================================================
:: VAP - CUDA Environment Setup v1.1.0
:: Location : <PROJECT_ROOT>\config\vap_setup_cuda.bat
:: Venv      : <REPO_PARENT>\.vap_env_cuda   (outside repo)
:: ============================================================
set "SCRIPT_VERSION=1.1.0"

:: --- Path resolution -----------------------------------------
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
for %%i in ("%SCRIPT_DIR%") do set "PROJECT_ROOT=%%~dpi"
set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"
for %%i in ("%PROJECT_ROOT%") do set "REPO_PARENT=%%~dpi"
set "REPO_PARENT=%REPO_PARENT:~0,-1%"

set "ENV_DIR=%REPO_PARENT%\.vap_env_cuda"
set "LOG_DIR=%PROJECT_ROOT%\logs"
set "FFMPEG_DIR=%PROJECT_ROOT%\ffmpeg"
set "TEMP_DIR=%PROJECT_ROOT%\_temp_ffmpeg"
set "REQ_FILTERED=%PROJECT_ROOT%\_req_cuda_filtered.txt"
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

:: Locale-safe timestamp
for /f "delims=" %%t in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TIMESTAMP=%%t"

:: --- Create dirs BEFORE first log write ----------------------
if not exist "%LOG_DIR%"              mkdir "%LOG_DIR%"
if not exist "%PROJECT_ROOT%\process" mkdir "%PROJECT_ROOT%\process"
if not exist "%PROJECT_ROOT%\models"  mkdir "%PROJECT_ROOT%\models"

set "LOG_FILE=%LOG_DIR%\setup_cuda_%TIMESTAMP%.log"
echo [%TIMESTAMP%] === VAP CUDA Setup v%SCRIPT_VERSION% === > "%LOG_FILE%"

:: ============================================================
:: ARGS
:: ============================================================
set "HF_TOKEN="
set "VOSK_SIZE=small"
set "SKIP_PYANNOTE="
set "SKIP_VOSK="
set "FORCE_CUDA_TAG="

:parse_args
if "%~1"=="" goto :args_done
if /i "%~1"=="--hf-token"      ( set "HF_TOKEN=%~2"       & shift & shift & goto :parse_args )
if /i "%~1"=="--vosk-size"     ( set "VOSK_SIZE=%~2"      & shift & shift & goto :parse_args )
if /i "%~1"=="--cuda-tag"      ( set "FORCE_CUDA_TAG=%~2" & shift & shift & goto :parse_args )
if /i "%~1"=="--skip-pyannote" ( set "SKIP_PYANNOTE=1"    & shift & goto :parse_args )
if /i "%~1"=="--skip-vosk"     ( set "SKIP_VOSK=1"        & shift & goto :parse_args )
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
echo  ^|     VAP CUDA Environment Setup v%SCRIPT_VERSION%      ^|
echo  +----------------------------------------------+
echo  Project root : %PROJECT_ROOT%
echo  Venv (CUDA)  : %ENV_DIR%
echo  Log          : %LOG_FILE%
echo.
echo  Usage: vap_setup_cuda.bat [--hf-token TOKEN]
echo                            [--vosk-size small^|large]
echo                            [--cuda-tag cu118^|cu121^|cu124]
echo                            [--skip-pyannote] [--skip-vosk]
echo.

:: ============================================================
:: STEP 1 - GPU and CUDA detection
:: ============================================================
call :header "STEP 1/6" "GPU and CUDA detection"

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    call :log "ERROR" "nvidia-smi not found"
    echo  [FATAL] nvidia-smi not found.
    echo          Install NVIDIA drivers: https://www.nvidia.com/drivers
    pause & exit /b 1
)

:: Query structured GPU info via nvidia-smi CSV (no pipe/findstr needed)
for /f "delims=" %%v in ('powershell -NoProfile -Command "(nvidia-smi --query-gpu=name --format=csv,noheader).Trim()"') do set "GPU_NAME=%%v"
for /f "delims=" %%v in ('powershell -NoProfile -Command "(nvidia-smi --query-gpu=memory.total --format=csv,noheader).Trim()"') do set "GPU_VRAM=%%v"
for /f "delims=" %%v in ('powershell -NoProfile -Command "(nvidia-smi --query-gpu=driver_version --format=csv,noheader).Trim()"') do set "DRIVER_VER=%%v"

:: CUDA version is only in the text output.
:: Avoid complex regex groups inside batch quotes -- use -replace chain instead.
for /f "delims=" %%v in ('powershell -NoProfile -Command "(nvidia-smi | Where-Object { $_ -match 'CUDA Version' }) -replace '.*CUDA Version:\s*','' -replace '\s.*',''"') do set "CUDA_VER_RAW=%%v"

if "%CUDA_VER_RAW%"=="" (
    call :log "ERROR" "Could not parse CUDA version from nvidia-smi"
    echo  [FATAL] Could not detect CUDA version from nvidia-smi output.
    echo          Force a tag manually: vap_setup_cuda.bat --cuda-tag cu124
    pause & exit /b 1
)

for /f "tokens=1,2 delims=." %%a in ("%CUDA_VER_RAW%") do (
    set "CUDA_MAJOR=%%a"
    set "CUDA_MINOR=%%b"
)

call :log "INFO" "GPU          : %GPU_NAME%"
call :log "INFO" "VRAM         : %GPU_VRAM%"
call :log "INFO" "Driver ver   : %DRIVER_VER%"
call :log "INFO" "CUDA version : %CUDA_VER_RAW%"
echo  [GPU] %GPU_NAME% / %GPU_VRAM% / Driver %DRIVER_VER% / CUDA %CUDA_VER_RAW%

:: Manual override
if not "%FORCE_CUDA_TAG%"=="" (
    set "CUDA_TAG=%FORCE_CUDA_TAG%"
    call :log "INFO" "CUDA tag forced: %CUDA_TAG%"
    goto :cuda_tag_set
)

:: Map CUDA version to torch wheel tag
:: torch 2.6.0 supports cu124, cu121, cu118
set "CUDA_TAG="
if !CUDA_MAJOR! GEQ 12 (
    if !CUDA_MINOR! GEQ 4 ( set "CUDA_TAG=cu124" & goto :cuda_tag_set )
    if !CUDA_MINOR! GEQ 1 ( set "CUDA_TAG=cu121" & goto :cuda_tag_set )
    set "CUDA_TAG=cu121"
    call :log "WARN" "CUDA 12.0 detected, using cu121 (closest compatible)"
    goto :cuda_tag_set
)
if !CUDA_MAJOR! EQU 11 (
    if !CUDA_MINOR! GEQ 8 ( set "CUDA_TAG=cu118" & goto :cuda_tag_set )
)

call :log "ERROR" "CUDA %CUDA_VER_RAW% not supported. torch 2.6.0 requires CUDA 11.8+"
echo  [FATAL] CUDA %CUDA_VER_RAW% is too old. Requires CUDA 11.8+.
echo          Update drivers: https://www.nvidia.com/drivers
pause & exit /b 1

:cuda_tag_set
set "TORCH_INDEX=https://download.pytorch.org/whl/%CUDA_TAG%"
call :log "OK" "CUDA %CUDA_VER_RAW% -> wheel tag: %CUDA_TAG%"
call :log "OK" "Torch index: %TORCH_INDEX%"

:: ============================================================
:: STEP 2 - Detect Python 3.11
:: ============================================================
call :header "STEP 2/6" "Detecting Python 3.11"

set "PYTHON_EXE="
set "PY_VER="

where py >nul 2>&1
if not errorlevel 1 (
    py -3.11 -c "import sys; v=sys.version_info; sys.exit(0 if v.major==3 and v.minor==11 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=py -3.11"
        for /f "tokens=*" %%v in ('py -3.11 -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
        goto :python_found
    )
    call :log "WARN" "py.exe found but not 3.11"
)

where python >nul 2>&1
if not errorlevel 1 (
    python -c "import sys; v=sys.version_info; sys.exit(0 if v.major==3 and v.minor==11 else 1)" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
        for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
        goto :python_found
    )
    for /f "tokens=*" %%v in ('python -c "import sys; print(sys.version.split()[0])" 2^>nul') do set "_FV=%%v"
    call :log "WARN" "python found but version is !_FV! (need 3.11)"
)

where python3.11 >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_EXE=python3.11"
    for /f "tokens=*" %%v in ('python3.11 -c "import sys; print(sys.version.split()[0])"') do set "PY_VER=%%v"
    goto :python_found
)

call :log "ERROR" "Python 3.11 not found"
echo  [FATAL] Python 3.11 not found. Install from https://python.org
pause & exit /b 1

:python_found
call :log "OK" "Python %PY_VER% via [%PYTHON_EXE%]"

:: ============================================================
:: STEP 3 - Virtual environment
:: ============================================================
call :header "STEP 3/6" "Virtual environment (CUDA)"

set "VENV_OK=0"
if exist "%ENV_DIR%\Scripts\python.exe" (
    if exist "%ENV_DIR%\Scripts\activate.bat" (
        "%ENV_DIR%\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>&1
        if not errorlevel 1 set "VENV_OK=1"
    )
)

if "!VENV_OK!"=="0" (
    if exist "%ENV_DIR%" (
        call :log "WARN" "Broken/partial venv at %ENV_DIR% -- removing"
        rmdir /s /q "%ENV_DIR%" >> "%LOG_FILE%" 2>&1
        if exist "%ENV_DIR%" (
            call :log "ERROR" "Cannot remove %ENV_DIR% -- close any processes using it"
            pause & exit /b 1
        )
    )
    call :log "INFO" "Creating venv at %ENV_DIR%"
    %PYTHON_EXE% -m venv "%ENV_DIR%" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log "ERROR" "venv creation failed"
        pause & exit /b 1
    )
    call :log "OK" "Venv created"
) else (
    call :log "SKIP" "Venv OK -- reusing"
)

call "%ENV_DIR%\Scripts\activate.bat"

"%ENV_DIR%\Scripts\python.exe" -c "import sys; sys.exit(0)" >nul 2>&1
if errorlevel 1 (
    call :log "ERROR" "Venv python.exe not functional after activation"
    pause & exit /b 1
)

set "VENV_PY=%ENV_DIR%\Scripts\python.exe"
set "VENV_PIP=%ENV_DIR%\Scripts\pip.exe"

for /f "delims=" %%p in ('"%VENV_PY%" -c "import sys; print(sys.executable)"') do set "ACTIVE_PY=%%p"
call :log "OK" "Venv active: !ACTIVE_PY!"

:: ============================================================
:: STEP 4 - Dependencies
:: ============================================================
call :header "STEP 4/6" "Python dependencies (CUDA build)"

call :log "INFO" "Upgrading pip/setuptools/wheel"
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel >> "%LOG_FILE%" 2>&1
if errorlevel 1 ( call :log "ERROR" "pip upgrade failed" & pause & exit /b 1 )

:: Install torch CUDA wheels FIRST with exclusive index-url.
:: This prevents pip from later resolving a CPU wheel from requirements.txt.
call :log "INFO" "Installing torch==2.6.0+%CUDA_TAG% and torchaudio==2.6.0+%CUDA_TAG%"
"%VENV_PIP%" install torch==2.6.0+%CUDA_TAG% torchaudio==2.6.0+%CUDA_TAG% --index-url "%TORCH_INDEX%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "ERROR" "torch CUDA install failed"
    echo  [FATAL] torch CUDA wheel install failed.
    echo          Try a different tag: vap_setup_cuda.bat --cuda-tag cu121
    pause & exit /b 1
)
call :log "OK" "torch 2.6.0+%CUDA_TAG% installed"

:: Filter requirements.txt -- remove lines that would conflict:
::   --extra-index-url .../cpu   (would re-pull CPU torch)
::   torch==                     (already installed)
::   torchaudio==                (already installed)
::   onnxruntime==               (replaced by onnxruntime-gpu)
set "REQ_FILE=%SCRIPT_DIR%\requirements.txt"
if not exist "%REQ_FILE%" set "REQ_FILE=%PROJECT_ROOT%\requirements.txt"
if not exist "%REQ_FILE%" (
    call :log "ERROR" "requirements.txt not found"
    pause & exit /b 1
)
call :log "INFO" "Filtering requirements.txt -> %REQ_FILTERED%"
findstr /v /r /c:"^--extra-index-url" /c:"^torch==" /c:"^torchaudio==" /c:"^# torchvision" /c:"^onnxruntime==" "%REQ_FILE%" > "%REQ_FILTERED%"

:: onnxruntime-gpu 1.18.0 supports CUDA 11.8 and 12.x
call :log "INFO" "Installing onnxruntime-gpu==1.18.0"
"%VENV_PIP%" install onnxruntime-gpu==1.18.0 >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "WARN" "onnxruntime-gpu==1.18.0 failed, trying latest"
    "%VENV_PIP%" install onnxruntime-gpu >> "%LOG_FILE%" 2>&1
    if errorlevel 1 call :log "WARN" "onnxruntime-gpu install failed"
) else (
    call :log "OK" "onnxruntime-gpu==1.18.0 installed"
)

call :log "INFO" "Installing webrtcvad-wheels"
"%VENV_PIP%" install webrtcvad-wheels >> "%LOG_FILE%" 2>&1
if errorlevel 1 call :log "WARN" "webrtcvad-wheels failed -- source build will be attempted"

call :log "INFO" "pip install -r (filtered requirements)"
"%VENV_PIP%" install -r "%REQ_FILTERED%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "ERROR" "pip install (filtered) failed. Check %LOG_FILE%"
    pause & exit /b 1
)
del "%REQ_FILTERED%" >nul 2>&1
call :log "OK" "Dependencies installed"

:: ============================================================
:: STEP 5 - GPU validation
:: ============================================================
call :header "STEP 5/6" "Validating GPU access"

call :log "INFO" "Checking torch.cuda.is_available()"
"%VENV_PY%" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" >nul 2>&1
if errorlevel 1 (
    call :log "ERROR" "torch.cuda.is_available() = False"
    echo  [FATAL] torch installed but CUDA not accessible.
    echo.
    echo  Diagnosis:
    echo    Driver CUDA : %CUDA_VER_RAW%   Wheel tag: %CUDA_TAG%
    echo    1. Version mismatch -- try a different tag: --cuda-tag cu121
    echo    2. cuDNN missing: https://developer.nvidia.com/cudnn
    echo    3. CPU torch still shadowing CUDA torch in venv:
    echo       Run: "%VENV_PIP%" uninstall torch torchaudio -y
    echo       Then re-run this script.
    pause & exit /b 1
)
call :log "OK" "torch.cuda.is_available() = True"

for /f "delims=" %%g in ('"%VENV_PY%" -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)} / {torch.cuda.get_device_properties(i).total_memory//1024**2} MB') for i in range(torch.cuda.device_count())]"') do (
    call :log "OK" "%%g"
    echo  [OK] %%g
)

"%VENV_PY%" -c "import onnxruntime as ort; assert 'CUDAExecutionProvider' in ort.get_available_providers()" >nul 2>&1
if errorlevel 1 (
    call :log "WARN" "onnxruntime CUDAExecutionProvider not available -- will use CPU"
) else (
    call :log "OK" "onnxruntime CUDAExecutionProvider available"
)

:: ============================================================
:: STEP 6 - ffmpeg + Models
:: ============================================================
call :header "STEP 6/6" "ffmpeg and models"

:: ffmpeg
where ffmpeg >nul 2>&1
if not errorlevel 1 (
    for /f "tokens=*" %%p in ('where ffmpeg') do call :log "SKIP" "ffmpeg on PATH: %%p"
    goto :ffmpeg_done
)
if exist "%FFMPEG_DIR%\ffmpeg.exe" goto :ffmpeg_set_path

if not exist "%TEMP_DIR%" mkdir "%TEMP_DIR%"
call :log "INFO" "Downloading ffmpeg"
powershell -NoProfile -Command "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%TEMP_DIR%\ffmpeg.zip' -UseBasicParsing" >> "%LOG_FILE%" 2>&1
if errorlevel 1 ( call :log "ERROR" "ffmpeg download failed" & pause & exit /b 1 )
powershell -NoProfile -Command "Expand-Archive -Path '%TEMP_DIR%\ffmpeg.zip' -DestinationPath '%TEMP_DIR%' -Force" >> "%LOG_FILE%" 2>&1
if not exist "%FFMPEG_DIR%" mkdir "%FFMPEG_DIR%"
for /D %%f in ("%TEMP_DIR%\ffmpeg-*") do (
    xcopy /E /I /Y "%%f\bin\*" "%FFMPEG_DIR%\" >> "%LOG_FILE%" 2>&1
    goto :ffmpeg_copied
)
:ffmpeg_copied
rmdir /s /q "%TEMP_DIR%"
call :log "OK" "ffmpeg extracted to %FFMPEG_DIR%"

:ffmpeg_set_path
for /f "skip=2 tokens=3*" %%a in ('reg query HKCU\Environment /v PATH 2^>nul') do set "_UP=%%a %%b"
echo !_UP! | find /i "%FFMPEG_DIR%" >nul 2>&1
if errorlevel 1 (
    setx PATH "!_UP!;%FFMPEG_DIR%" >> "%LOG_FILE%" 2>&1
    call :log "OK" "ffmpeg added to user PATH permanently"
) else (
    call :log "SKIP" "ffmpeg already in user PATH"
)
set "PATH=%FFMPEG_DIR%;%PATH%"
:ffmpeg_done
call :log "OK" "ffmpeg ready"

:: .env
if not exist "%PROJECT_ROOT%\.env" ( echo # VAP config > "%PROJECT_ROOT%\.env" )

if not "%HF_TOKEN%"=="" (
    findstr /i "HF_TOKEN" "%PROJECT_ROOT%\.env" >nul 2>&1
    if errorlevel 1 ( echo HF_TOKEN=%HF_TOKEN% >> "%PROJECT_ROOT%\.env" & call :log "OK" "HF_TOKEN -> .env" )
    else ( call :log "SKIP" "HF_TOKEN already in .env" )
) else (
    call :log "WARN" "No --hf-token. pyannote models will be skipped."
    call :log "WARN" "Re-run with: vap_setup_cuda.bat --hf-token hf_xxxxxxxxxx"
)

findstr /i "CUDA_TAG" "%PROJECT_ROOT%\.env" >nul 2>&1
if errorlevel 1 ( echo CUDA_TAG=%CUDA_TAG% >> "%PROJECT_ROOT%\.env" )
findstr /i "VAP_DEVICE" "%PROJECT_ROOT%\.env" >nul 2>&1
if errorlevel 1 ( echo VAP_DEVICE=cuda >> "%PROJECT_ROOT%\.env" )

:: Models
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
if errorlevel 1 call :log "WARN" "_setup_models.py had errors -- check console above"

:: ============================================================
:: DONE
:: ============================================================
echo.
echo  +----------------------------------------------+
echo  ^|           CUDA Setup complete!               ^|
echo  +----------------------------------------------+
echo.
echo   Root      : %PROJECT_ROOT%
echo   Venv      : %ENV_DIR%
echo   Log       : %LOG_FILE%
echo   CUDA tag  : %CUDA_TAG%  (torch 2.6.0+%CUDA_TAG%)
echo   .env      : %PROJECT_ROOT%\.env  [VAP_DEVICE=cuda]
echo   Models    : %PROJECT_ROOT%\models\
echo.
echo   Activate in a new terminal:
echo     call "%ENV_DIR%\Scripts\activate.bat"
echo.
if "%HF_TOKEN%"=="" (
    echo  [!] pyannote NOT downloaded - no HF token.
    echo      Re-run: vap_setup_cuda.bat --hf-token hf_xxxxxxxxxx
    echo.
)
call :log "INFO" "=== CUDA Setup finished ==="
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