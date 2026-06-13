# Build/run the API with uv inside the container.
FROM python:3.12-slim

# Tesseract (OCR) + OpenCV runtime libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the uv binary from its official image (no pip needed).
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy the project and install runtime deps into an in-project .venv.
# --no-dev skips pytest/httpx; --compile-bytecode speeds up container start.
COPY . .
RUN uv sync --no-dev --compile-bytecode

# Run the app through uv so it uses the managed environment.
EXPOSE 8000
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
