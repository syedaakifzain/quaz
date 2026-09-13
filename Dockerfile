FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "qiskit>=2.0" \
    "qiskit-machine-learning>=0.9.0" \
    "mne>=1.6" \
    "numpy>=1.26" \
    "scipy>=1.12" \
    "PyWavelets>=1.5" \
    "scikit-learn>=1.4" \
    "pandas>=2.1" \
    "matplotlib>=3.8" \
    "reportlab>=4.0" \
    "pyyaml>=6.0" \
    "joblib>=1.3" \
    "dill>=0.3.8" \
    "fastapi" \
    "uvicorn[standard]" \
    "python-multipart"

COPY api ./api
COPY config ./config
COPY core ./core
COPY eeg_processing ./eeg_processing
COPY inference ./inference
COPY quantum ./quantum
COPY training ./training
COPY models ./models
COPY test_data ./test_data

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]