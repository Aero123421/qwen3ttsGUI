@echo off
setlocal

cd /d %~dp0

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo [.env] を .env.example から作成しました。
)

echo Docker Compose で Qwen3-TTS GUI を起動します...
docker compose up --build -d

if errorlevel 1 (
    echo 起動に失敗しました。Docker Desktop と NVIDIA GPU 設定を確認してください。
    exit /b 1
)

echo 起動しました。ブラウザで http://localhost:7860 を開いてください。
docker compose ps

endlocal
