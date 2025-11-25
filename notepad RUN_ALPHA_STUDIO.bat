@echo off
chcp 65001 >nul
title ALPHA IMAGE STUDIO - Start
echo.
echo ================================
echo   ALPHA IMAGE STUDIO - START
echo ================================
echo.

if not exist .venv (
    echo [BŁĄD] Brak folderu .venv
    echo Najpierw uruchom: INSTALL_ALPHA_STUDIO.bat
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

python alpha_studio_ui.py

pause

