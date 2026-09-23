FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml LICENSE ./
COPY ddsim ./ddsim
RUN pip install --no-cache-dir ".[serve]"

EXPOSE 8080
CMD ["ddsim", "serve", "--host", "0.0.0.0", "--port", "8080"]
