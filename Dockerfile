FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        gfortran \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install ".[foundation,test]"

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests

VOLUME ["/app/data", "/app/results"]

ENTRYPOINT ["python"]
CMD ["scripts/run_study.py", "--config", "configs/study.yaml", "--data", "data", "--output", "results/main"]
