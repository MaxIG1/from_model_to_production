# Use the official Python image as the base
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the local code and data to the container
COPY . /app

# Install necessary moduls
RUN pip install Flask && \
    pip install scikit-learn==1.3.0 && \
    pip install mlflow==2.6.0 && \
    pip install pandas==2.1.1 && \
    pip install boto3 && \
    pip install datetime && \
    pip install psycopg2-binary && \
    pip install pathlib==1.0.1 && \
    pip install numpy==1.26  && \
    pip install psutil==5.9.0 && \
    pip install datetime && \
    pip install statsmodels && \
    pip install flask_httpauth && \
    pip install apscheduler && \
    pip install python-dotenv



# Expose port 5001
EXPOSE 5001

# Specify the command to run your Flask application
CMD python3 ./cloud_fraud_detection.py
