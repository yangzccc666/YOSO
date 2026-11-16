@echo off
setlocal
chcp 65001 >nul
title YOLO Environment Setup
set "YOLO_SETUP_CONDA="
if defined CONDA_EXE if exist "%CONDA_EXE%" set "YOLO_SETUP_CONDA=%CONDA_EXE%"
if not defined YOLO_SETUP_CONDA for /f "delims=" %%I in ('where conda.exe 2^>nul') do if not defined YOLO_SETUP_CONDA set "YOLO_SETUP_CONDA=%%I"
if not defined YOLO_SETUP_CONDA if defined CONDA_PREFIX if exist "%CONDA_PREFIX%\Scripts\conda.exe" set "YOLO_SETUP_CONDA=%CONDA_PREFIX%\Scripts\conda.exe"
for %%D in ("%USERPROFILE%\anaconda3" "%USERPROFILE%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%ProgramData%\anaconda3" "%ProgramData%\miniconda3") do if not defined YOLO_SETUP_CONDA if exist "%%~D\Scripts\conda.exe" set "YOLO_SETUP_CONDA=%%~D\Scripts\conda.exe"
if not defined YOLO_SETUP_CONDA (
  echo Cannot find Anaconda or Miniconda.
  echo Open Anaconda Prompt, enter this folder, then run install_yolo_env.bat
  echo Do not install into the base environment.
  pause
  exit /b 1
)
echo Conda: "%YOLO_SETUP_CONDA%"
echo Internet access is required. Downloads may take several minutes.
"%YOLO_SETUP_CONDA%" run --no-capture-output -n base python "%~dp0install_yolo_env.py" --conda "%YOLO_SETUP_CONDA%" %*
set "YOLO_SETUP_RESULT=%ERRORLEVEL%"
if not "%YOLO_SETUP_RESULT%"=="0" echo Setup failed. Read the error above; no existing environment was deleted.
pause
exit /b %YOLO_SETUP_RESULT%
