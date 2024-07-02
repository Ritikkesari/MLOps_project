# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port number that the FastAPI application should run on
EXPOSE 8000

# Define environment variables
ENV MLFLOW_TRACKING_URI="sqlite:///02-experiment_tracking/mlflow.db"

# Command to run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
