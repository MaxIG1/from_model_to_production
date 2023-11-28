# from_model_to_production
# Project Overview



This project demonstrates a Flask-based web application for fraud detection that utilizes MLflow for model management. The application is designed to predict whether a person might intend to commit fraud based on input features. It includes functionality for initial training, retraining, and prediction with model versioning and data drift detection. See the next picture for an project overview.

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
   - Start the MLflow server in the background of your EC2 server:
     ```bash
     mlflow server -h 0.0.0.0 -p 5000 --app-name basic-auth --backend-store-uri [your_postgress_path] --default-artifact-root [your_s3_store] > mlflow_server.log 2>&1 &
     ```

4. **Credentials File:**
   - Create a file named `credentials.env` with the necessary credentials for PostgreSQL, MLflow Tracking Server, and MLflow UI access. A example file is provided

5. **Run the Application:**
   - Access the Flask application at `http://docker_server_ip:5001` in your web browser or via your prefered programm via the post method.

## Application Usage

- **Endpoints:**
  - `/hello`: Test endpoint to check if Flask is running.
  - `/initialtrain`: Manually trigger the initial training of the machine learning model.
  - `/retrain`: Manually trigger the retraining of the machine learning model.
  - `/predict`: Endpoint to make predictions. Supports both GET and POST requests.

    -example of an GET way via webbrowser http://docker_server_ip:5001/predict?account_balance=4&duration_of_credit_month=12&payment_status_of_previous_credit=4&purpose=3&credit_amount=618&value_savings_stocks=1&length_of_current_employment=5&instalment_per_cent=4&sex_marital_status=3&guarantors=1&duration_in_current_address=4&most_valuable_available_asset=2&age_years=56&concurrent_credits=3&type_of_apartment=2&no_of_credits_at_this_bank=1&occupation=1&no_of_dependents=3&telephone=0&foreign_worker=0

    -example of an POST way see post_request.py

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
