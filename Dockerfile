FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    GRADIO_ANALYTICS_ENABLED=False

WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-cpu.txt ./
RUN python -m pip install -r requirements-cpu.txt

COPY app.py ./
COPY src ./src
COPY ui ./ui
COPY configs ./configs
COPY data ./data
COPY artifacts ./artifacts

# Cache the original encoder and retrieval vectors in the server image.
# This reads the saved head; it does not train or change model settings.
RUN python -c "from src.pipeline import TicketAssistant; TicketAssistant(use_llm=False)"

EXPOSE 7860
CMD ["sh", "-c", "exec python app.py --host 0.0.0.0 --port ${PORT:-7860}"]
