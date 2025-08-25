# Simple Dockerfile to run the Streamlit KML/KMZ altitude app
FROM python:3.11-slim

# Prevents Python from buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1     PYTHONUNBUFFERED=1     PIP_NO_CACHE_DIR=1     PORT=8501

# System deps (curl for health checks, build tools for any wheels if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY kmz_altitude_app.py ./

# Streamlit tweaks (no telemetry, headless)
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501
CMD ["streamlit", "run", "kmz_altitude_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
