# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files (including models) into image
# using a single COPY ensures the weights and source files
# are all present for deployment.
COPY . .

# Set environment variable for production
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Expose port for the web service
EXPOSE 8000

# Use gunicorn to serve the Flask app in production
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} app:app"]
