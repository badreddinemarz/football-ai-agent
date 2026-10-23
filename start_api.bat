@echo off
rem Starts the REST API with Claude Haiku and opens its interactive page in the browser.
rem Needs ANTHROPIC_API_KEY in .env. Close this window to stop the API.
cd /d "%~dp0"
set LLM_MODEL=claude-haiku-5-5
start "" cmd /c "timeout /t 15 >nul & start http://127.0.0.1:8000/docs"
.venv\Scripts\python -m uvicorn app.api:app --port 8000
