# Build stage
FROM python:3.10-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.10-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
EXPOSE 8081

CMD ["python", "phone_rag.py", "start"]

# FROM python:3.10-slim

# RUN apt-get update && apt-get install -y gcc python3-dev && rm -rf /var/lib/apt/lists/*
 

# WORKDIR /app

# # Copy requirements and install dependencies
# COPY requirements.txt .
# RUN pip install --upgrade pip
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy your project files
# COPY . .

# # Expose any necessary ports (if your service exposes health check endpoints, for example)
# EXPOSE 8081

# CMD ["python", "phone_rag.py", "start"]
