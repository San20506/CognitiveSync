FROM python:3.11-slim

WORKDIR /app

RUN pip install uv --quiet

COPY pyproject.toml uv.lock ./

# Install CPU-only torch — bypasses the CUDA packages in uv.lock
RUN pip install \
    torch==2.2.0+cpu \
    torch-geometric==2.5.3 \
    --index-url https://download.pytorch.org/whl/cpu \
    --quiet

# Export all remaining deps from lockfile, skipping torch (already installed)
# This avoids uv sync pulling CUDA torch from the lockfile
RUN uv export --no-dev --frozen \
    --no-emit-package torch \
    --no-emit-package torch-geometric \
    --no-emit-package torchvision \
    --no-emit-package torchaudio \
    -o /tmp/requirements.txt \
    && pip install -r /tmp/requirements.txt --quiet

# Copy source (after deps — layer cache hits on rebuild)
COPY api/ api/
COPY intelligence/ intelligence/
COPY ingestion/ ingestion/
COPY config/ config/
COPY output/ output/
COPY alembic/ alembic/
COPY alembic.ini .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
