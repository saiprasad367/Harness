# Open-Turn Voice Lab — Frontend Architecture

## Project Overview
This frontend is a React 18 + TanStack Router + Tailwind CSS application representing the research workspace for:
**“An Adaptive, Accessibility-Aware Agent Harness for Full-Duplex Voice Models.”**

## Connected Research Services
- **Backend**: FastAPI / Starlette server at `api_server.py` (`http://localhost:8000` / `/ws/metrics` / `/ws/voice`)
- **Voice Engine**: Genuine Kyutai Moshi 7B on Kaggle GPU / local WebSocket endpoint
- **Dataset**: TORGO 8,218 clean head-mic corpus with 15 disjoint speakers
