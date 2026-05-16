@echo off
echo Compilando frontend React...
cd /d "%~dp0frontend"
call npm install
call npm run build
echo.
echo Build completado. Archivos en: dashboard\frontend\dist\
echo El backend sirve el frontend automaticamente desde ese directorio.
pause
