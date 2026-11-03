# The REST API (app/api.py) in a container.
#
#   docker build -t football-agent .
#   docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" football-agent
#   → http://127.0.0.1:8000/docs
#
# data/ (DuckDB database, Chroma index) is not in the image: build it once with the scripts/ and mount it.
# Secrets (ANTHROPIC_API_KEY...) are given at run time with --env-file, never written into the image.

FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# CPU-only torch first: the default Linux wheel ships CUDA, several GB that are useless without a GPU
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies before the code: Docker caches this layer, so changing the code does not reinstall them
COPY requirements.txt .
RUN pip install -r requirements.txt

# Download the embedding model now, at build time, instead of at every container start
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-base')"

COPY src/ src/
COPY app/ app/

# Defaults, overridable with -e at run time (e.g. -e AGENT=handwritten)
ENV LLM_MODEL=claude-haiku-5-5 AGENT=langgraph

EXPOSE 8000
# 0.0.0.0: listen to connections coming from outside the container (127.0.0.1 would only accept the container itself)
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
