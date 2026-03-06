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
EXPOSE 7860

# Use uvicorn to serve the FastAPI app in production
CMD ["uvicorn", "--host", "0.0.0.0", "--port", "7860", "app:app"]
