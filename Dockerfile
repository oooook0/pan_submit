FROM nvcr.io/nvidia/cuda:12.8.0-cudnn-devel-ubuntu24.04

# Install dependencies
RUN apt-get update && apt-get install -y git python3-pip

# Set working directory
WORKDIR /app

# Copy files
COPY . .

# Install Python dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    git curl build-essential libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt --break-system-packages --no-cache
RUN pip config set global.break-system-packages true \
	&& python3 -m spacy download en_core_web_sm \
	&& python3 -c "import nltk; nltk.download('wordnet')"

# Define entrypoint
ENTRYPOINT ["python3", "run.py"]
