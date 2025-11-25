@echo off
chcp 65001 >nul
title ALPHA IMAGE STUDIO - Instalacja
echo.
echo ================================
echo   ALPHA IMAGE STUDIO - INSTALL
echo ================================
echo.

REM Tworzymy wirtualne srodowisko .venv
if not exist .venv (
    echo [1/3] Tworze wirtualne srodowisko (.venv)...
    python -m venv .venv
) else (
    echo [1/3] Wirtualne srodowisko juz istnieje (.venv) - pomijam
)

echo [2/3] Aktywuje srodowisko i aktualizuje pip...
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip

echo [3/3] Instalacja pakietow z requirements.txt...
pip install -r requirements.txt

echo.
echo ==========================================
echo   INSTALACJA ZAKONCZONA POWODZENIEM
echo ==========================================
echo.
echo Aby uruchomic studio, uruchom:
echo    RUN_ALPHA_STUDIO.bat
echo.
pause
