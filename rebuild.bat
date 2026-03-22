@echo off
setlocal

cd /d "%~dp0"

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [INFO] Created .env from .env.example
)

echo [INFO] Rebuilding Qwen3-TTS GUI image with Docker Compose...
docker compose up --build -d

if errorlevel 1 (
    echo [ERROR] Failed to rebuild Docker Compose.
    echo [HINT] Check Docker Desktop, docker compose, and NVIDIA GPU settings.
    pause
    exit /b 1
)

echo [INFO] Rebuild completed.
docker compose ps

endlocal
