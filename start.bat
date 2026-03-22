@echo off
setlocal

cd /d "%~dp0"

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [INFO] Created .env from .env.example
)

set "PORT=7860"
for /f "tokens=2 delims==" %%A in ('findstr /b "GRADIO_SERVER_PORT=" ".env" 2^>nul') do set "PORT=%%A"

echo [INFO] Starting Qwen3-TTS GUI with Docker Compose...
docker compose up --build -d

if errorlevel 1 (
    echo [ERROR] Failed to start Docker Compose.
    echo [HINT] Check Docker Desktop, docker compose, and NVIDIA GPU settings.
    pause
    exit /b 1
)

echo [INFO] Waiting for http://localhost:%PORT% ...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$port='%PORT%'; $ok=$false; for($i=0; $i -lt 30; $i++){ try { $r = Invoke-WebRequest -Uri ('http://localhost:' + $port) -UseBasicParsing -TimeoutSec 2; if($r.StatusCode -ge 200){ $ok=$true; break } } catch {}; Start-Sleep -Seconds 1 }; if($ok){ exit 0 } else { exit 1 }"

if errorlevel 1 (
    echo [WARN] The container started, but the web UI did not respond yet.
    echo [HINT] Run: docker compose logs --tail 200
    docker compose ps
    pause
    exit /b 1
)

echo [INFO] Qwen3-TTS GUI is ready: http://localhost:%PORT%
docker compose ps
start "" "http://localhost:%PORT%"

endlocal
