FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install ripgrep (rg) and CA certificates for HTTPS API & web requests
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ripgrep \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md LICENSE server.py ./

# Run as a non-root user
RUN useradd --create-home --shell /bin/bash mcpuser \
    && chown -R mcpuser:mcpuser /app
USER mcpuser

ENTRYPOINT ["python", "/app/server.py"]
