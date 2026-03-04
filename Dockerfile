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

# Expose port for the web service
EXPOSE 8000

# Set environment variable for production
ENV PYTHONUNBUFFERED=1

# Use gunicorn to serve the Flask app in production
# binding to the same port that is exposed above.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]
