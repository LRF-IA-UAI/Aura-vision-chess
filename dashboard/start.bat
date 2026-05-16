@echo off
setlocal

echo.
echo  ==========================================
echo   AURA Dashboard — Inicio rapido
echo   Robot Ajedrecista - CAETI UAI
echo  ==========================================
echo.

:: Detectar IP local para mostrar la URL de LAN
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "IPv4" ^| findstr /v "127.0.0.1" ^| head -1') do (
    set "LOCAL_IP=%%a"
)
set LOCAL_IP=%LOCAL_IP: =%

:: 1 - Arrancar Mosquitto MQTT
echo [1/3] Iniciando Mosquitto MQTT...
where mosquitto >nul 2>&1
if %errorlevel% == 0 (
    start "AURA — Mosquitto" /min mosquitto -c "%~dp0mosquitto\mosquitto.conf"
    timeout /t 2 /nobreak >nul
    echo       OK — Mosquitto corriendo en 0.0.0.0:1883
) else (
    echo       WARN — 'mosquitto' no encontrado en PATH.
    echo              Instalar desde: https://mosquitto.org/download/
    echo              O agregar al PATH si ya esta instalado.
    echo              Continuando sin MQTT...
)

echo.

:: 2 - Instalar dependencias del backend si es necesario
echo [2/3] Verificando backend...
cd /d "%~dp0backend"
if not exist ".deps_ok" (
    echo       Instalando dependencias Python...
    pip install -r requirements.txt --quiet
    echo. > .deps_ok
)

:: 3 - Arrancar FastAPI backend
echo [3/3] Iniciando backend FastAPI (puerto 8000)...
start "AURA — Backend" cmd /k "cd /d "%~dp0backend" && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload"

echo.
echo  ==========================================
echo   Dashboard listo:
echo     Local:  http://localhost:8000
if defined LOCAL_IP (
    echo     Red:    http://%LOCAL_IP%:8000
)
echo.
echo   Para desarrollo con hot-reload:
echo     cd dashboard\frontend
echo     npm install ^&^& npm run dev
echo     Abrir: http://localhost:5173
echo  ==========================================
echo.
echo  Recordar: camera_pipeline.py debe estar corriendo.
echo.
pause
