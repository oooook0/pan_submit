FROM python:3.10-slim

# Install dependencies
RUN apt-get update && apt-get install -y git

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    git curl build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Define entrypoint
ENTRYPOINT ["python3", "run.py"]