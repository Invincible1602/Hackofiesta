# Use the official Python image as base
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y default-jdk && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download necessary NLTK data
RUN python -c "import nltk; nltk.download('stopwords')"

# Copy all project files to the container
COPY . .

# Expose the port FastAPI runs on (optional, for local use)
EXPOSE 8000

# Run FastAPI with the dynamic PORT from the environment variable, fallback to 8000 if not found
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
