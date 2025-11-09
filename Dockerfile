FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    git \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency files first for caching
COPY pyproject.toml uv.lock* requirements.txt ./

# Create and install dependencies
RUN uv venv .venv --clear && \
    uv pip install -r requirements.txt && \
    uv sync --no-dev

# Copy app code
COPY . .

EXPOSE 8080 80
CMD ["uv", "run", "--ip=0.0.0.0", "--allow-root"]