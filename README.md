# from_model_to_production
# Project Overview



This project demonstrates a Flask-based web application for fraud detection that utilizes MLflow for model management. The application is designed to predict whether a person might intend to commit fraud based on input features. It includes functionality for initial training, retraining, and prediction with model versioning and data drift detection.

![Alt text](<From model (1).png>)


## Prerequisites

To run this project, you need to set up two EC2 servers:
1. **Docker Server**: Responsible for running the Flask application and handling Docker containers.
2. **MLflow Server**: Hosts the MLflow server to manage and track machine learning experiments.
2.1. **Mlflow S3 Artifact Store**: Hosts the Models
2.2. **Mlflow Postgress Database**: Hosts the meta-data about the runs.  
3. **credentials.env**: Create a file with all you passwords

A good starting point for this task in an AWS solution is: Daryani, C. (May 28). MLflow on AWS: A Step-by-Step Setup Guide. AMA Technology Blog. https://medium.com/ama-tech-blog/mlflow-on-aws-a-step-by-step-setup-guide-8601414dd6ea


## Getting Started

1. **Clone the Repository:**
    ```bash
    git clone https://github.com/your_username/your_repository.git
    cd your_repository
    ```

2. **Setup Docker Server:**
   - Use the provided Dockerfile to set up the Docker environment on the Docker Server.
   - Ensure Docker is installed and running on the server.
   - Build the Docker image:
     ```bash
     docker build -t flask_ml_app .
     ```
   - Run the Docker container for example in EC2:
     ```bash
     docker container run -d -p 5001:5001 -e AWS_ACCESS_KEY_ID=access_key -e AWS_SECRET_ACCESS_KEY=secret_key container_name
     ```
A good starting point for an AWS solution is: Nagar, D. (2019, October 12). Running Docker on AWS EC2. AppGambit. Medium. https://medium.com/appgambit/part-1-running-docker-on-aws-ec2-cbcf0ec7c3f8



3. **Setup MLflow Server:**
   - Set up another EC2 server to act as the MLflow Server.
   - Install MLflow on the MLflow Server:
     ```bash
     pip install mlflow
     ```
   - Start the MLflow server:
     ```bash
     mlflow server --host 0.0.0.0 --port 5000
     ```

4. **Credentials File:**
   - Create a file named `credentials.env` with the necessary credentials for PostgreSQL, MLflow Tracking Server, and MLflow UI access.

5. **Run the Application:**
   - Access the Flask application at `http://docker_server_ip:5001` in your web browser.

## Application Usage

- **Endpoints:**
  - `/hello`: Test endpoint to check if Flask is running.
  - `/initialtrain`: Manually trigger the initial training of the machine learning model.
  - `/retrain`: Manually trigger the retraining of the machine learning model.
  - `/predict`: Endpoint to make predictions. Supports both GET and POST requests.

- **Authentication:**
  - Basic authentication is implemented. Use credentials from the PostgreSQL user table.

- **MLflow UI:**
  - Access MLflow UI at `http://mlflow_server_ip:5000` to track experiments and models.

- **Background Task:**
  - The application runs a background task for monthly model retraining.

## Additional Notes

- Ensure security group settings allow traffic to ports 5000 and 5001 on the respective servers.
- This is a simplified setup guide; adjust configurations based on your specific environment.

## Disclaimer

This project is intended for educational purposes, and deployment in a production environment may require additional security and optimization measures. Use it responsibly and adapt it according to your needs.
