#!/bin/sh
set -eu

# Allow compose / docker run overrides: `docker run IMAGE streamlit ...`
if [ "$#" -gt 0 ]; then
  exec "$@"
fi

streamlit run streamlit_app.py \
  --server.address=0.0.0.0 \
  --server.port=8501 \
  --server.headless=true &

exec uvicorn app.api:app --host 0.0.0.0 --port 8000
