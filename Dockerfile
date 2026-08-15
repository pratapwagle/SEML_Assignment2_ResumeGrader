FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements-runtime.txt

COPY app ./app
COPY ml ./ml
COPY services ./services
COPY quality ./quality
COPY models ./models
COPY report/metrics_snapshot.json ./report/metrics_snapshot.json
COPY .streamlit ./.streamlit
COPY config.py logging_config.py train.py run_api.py streamlit_app.py ./

RUN mkdir -p logs data \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
