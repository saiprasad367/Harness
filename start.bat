@echo off
echo ======================================================================
echo Starting Adaptive Voice Agent Harness with Kyutai Moshi
echo ======================================================================

echo Starting Python API Backend on http://localhost:8000...
start cmd /k "python api_server.py"

echo Starting React Frontend on http://localhost:3000...
start cmd /k "cd frontend\open-turn-voice-lab && npm.cmd run dev"

echo Done! Backend running on http://localhost:8000, Frontend running on http://localhost:3000 (or 5173)

