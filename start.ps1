# Start script for Adaptive Voice Agent Harness (Backend + Frontend)
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "🎙️ Starting Adaptive Voice Agent Harness with Kyutai Moshi" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# Start Python API Backend in a new window
Write-Host "Starting Python API Backend on http://localhost:8000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot'; python api_server.py"

# Start Frontend Dev Server in a new window
Write-Host "Starting React Frontend on http://localhost:3000..." -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PSScriptRoot\frontend\open-turn-voice-lab'; npm.cmd run dev"

Write-Host "Done! Backend running on http://localhost:8000, Frontend running on http://localhost:3000 (or 5173)" -ForegroundColor Yellow

