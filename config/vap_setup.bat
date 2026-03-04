@echo off
echo === Creating virtual environment ===
py -3.11 -m venv ".vap_env"

echo === Activating environment ===
call ".\.vap_env\Scripts\activate.bat"

echo === Installing Python requirements ===
python -m pip install --upgrade pip setuptools wheel
pip install -r "requirements.txt"

echo === Downloading NLTK stopwords ===
python -m nltk.downloader stopwords

echo === Downloading Spacy stopwords ===
python -m spacy download es_core_news_sm

echo === Installing ffmpeg ===
:: Set up variables (quoted!)
set "FFMPEG_URL=https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
set "TEMP_DIR=%CD%\temp_ffmpeg"
set "FFMPEG_DIR=%CD%\ffmpeg"

:: Make sure temp dir exists
mkdir "%TEMP_DIR%"
mkdir "%CD%\logs"
mkdir "%CD%\process"

:: Download ffmpeg (quoted PowerShell path)
powershell -Command "Invoke-WebRequest -Uri '%FFMPEG_URL%' -OutFile '%TEMP_DIR%\ffmpeg.zip'"

:: Unzip ffmpeg (quoted paths)
powershell -Command "Expand-Archive -Path '%TEMP_DIR%\ffmpeg.zip' -DestinationPath '%TEMP_DIR%'"

:: Move the /bin folder from unzipped contents to destination (careful with path detection)
for /D %%f in ("%TEMP_DIR%\ffmpeg-*") do (
    move "%%f\bin" "%FFMPEG_DIR%"
    goto :found
)
:found

:: Clean up temp
rmdir /s /q "%TEMP_DIR%"

:: Temporarily set PATH
set "PATH=%FFMPEG_DIR%;%PATH%"

echo === Setup complete! ===
pause