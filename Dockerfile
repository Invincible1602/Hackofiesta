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

# Expose the port FastAPI runs on
EXPOSE 8000

# Use the PORT environment variable that Render sets
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "0.0.0.0:$PORT"]

