@echo off
rem Starts the REST API with Claude Haiku and the LangGraph agent (with memory), and opens its interactive page in the browser.
rem Needs ANTHROPIC_API_KEY in .env. Close this window to stop the API.
cd /d "%~dp0"
set LLM_MODEL=claude-haiku-5-5
set AGENT=langgraph
start "" cmd /c "timeout /t 30 >nul & start http://127.0.0.1:8000/docs"
.venv\Scripts\python -m uvicorn app.api:app --port 8000
