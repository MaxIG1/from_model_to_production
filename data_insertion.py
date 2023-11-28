import psycopg2
import pandas as pd
import mlflow
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="credentials.env")

# Database connection parameters
db_host = os.getenv("DB_HOST")
db_name = os.getenv("DB_NAME")
db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")


# passwords and username for ml_flow server
MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

# setting the tracking server
TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")
# mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:5000")

# Construct the tracking URI with credentials
tracking_uri_with_credentials = f"http://{MLFLOW_TRACKING_USERNAME}:{MLFLOW_TRACKING_PASSWORD}@{TRACKING_SERVER_HOST}:5000"


# setting the tracking server
mlflow.set_tracking_uri(tracking_uri_with_credentials)


# Initialize your MLflow client
client = mlflow.tracking.MlflowClient()

# set the current version of the tracking
number = 51

# Specify the model name you want to retrieve
model_name = f"register_{number}"
latest_versions = client.get_latest_versions(name=model_name)


# deleting
drop_table_sql = "DROP TABLE IF EXISTS credit_table;"
try:
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=5432,  # The port for PostgreSQL
    )

    cursor = conn.cursor()
    print("Connected to the database!")

    # Create the table
    cursor.execute(drop_table_sql)
    print("Table dropped.")

    # Commit the changes
    conn.commit()

    # # Close the cursor and connection
    cursor.close()
    conn.close()


except Exception as e:
    print(f"Error: {e}")


# creating the database


create_table_sql = """CREATE TABLE credit_table (
    id SERIAL PRIMARY KEY,
    creditability INT,
    account_balance INT,
    duration_of_credit_month INT,
    payment_status_of_previous_credit INT,
    purpose INT,
    credit_amount INT,
    value_savings_stocks INT,
    length_of_current_employment INT,
    instalment_per_cent INT,
    sex_marital_status INT,
    guarantors INT,
    duration_in_current_address INT,
    most_valuable_available_asset INT,
    age_years INT,
    concurrent_credits INT,
    type_of_apartment INT,
    no_of_credits_at_this_Bank INT,
    occupation INT,
    no_of_dependents INT,
    telephone INT,
    foreign_worker INT
);"""


try:
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=5432,  # The port for PostgreSQL
    )

    cursor = conn.cursor()
    print("Connected to the database!")

    # Create the table
    cursor.execute(create_table_sql)
    print("Table created.")

    # Commit the changes
    conn.commit()

    # # Close the cursor and connection
    cursor.close()
    conn.close()


except Exception as e:
    print(f"Error: {e}")


# loading so that I get the columns right

try:
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=5432,  # The port for PostgreSQL
    )

    cursor = conn.cursor()
    print("Connected to the database!")

    # Select all values from the table
    cursor.execute("SELECT * FROM credit_table")
    rows = cursor.fetchall()

    column_list = [column[0] for column in cursor.description]
    df_data_credit = pd.DataFrame(rows, columns=column_list)
    df_data_credit.index = df_data_credit.id
    df_data_credit.sort_index(inplace=True)
    df_data_credit.drop(columns="id", inplace=True)

    # Close the cursor and connection
    cursor.close()
    conn.close()

except Exception as e:
    print(f"Error: {e}")


# Inserting the data

insert_data_sql = f"""
INSERT INTO credit_table ({", ".join(df_data_credit.columns)}) VALUES ({", ".join(["%s"] * len(df_data_credit.columns))});
"""

csv_file_path = "data/german_credit_resampled.csv"
# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path, index_col=[0])


try:
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=5432,  # The port for PostgreSQL
    )
    cursor = conn.cursor()

    # Define the table name
    table_name = "credit_table"

    # Loop through the DataFrame and insert data row by row
    for index, row in df.iterrows():
        values = tuple(row)
        cursor.execute(insert_data_sql, values)

    # Commit the changes and close the cursor and connection
    conn.commit()
    cursor.close()
    conn.close()
except Exception as e:
    print(e)


# add the new collumns

try:
    # Connect to the database
    conn = psycopg2.connect(
        host=db_host, database=db_name, user=db_user, password=db_password, port=5432
    )
    cursor = conn.cursor()
    print("Connected to the database!")

    cursor.execute("ALTER TABLE credit_table ADD COLUMN new_column BOOLEAN")
    cursor.execute("ALTER TABLE credit_table ADD COLUMN model_version INT")
    cursor.execute("UPDATE credit_table SET new_column = FALSE")

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

except Exception as e:
    print(f"Error: {e}")


# reloading the columns

try:
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=5432,  # The port for PostgreSQL
    )

    cursor = conn.cursor()
    print("Connected to the database!")

    # Select all values from the table
    cursor.execute("SELECT * FROM credit_table")
    rows = cursor.fetchall()

    column_list = [column[0] for column in cursor.description]
    df_data_credit = pd.DataFrame(rows, columns=column_list)
    df_data_credit.index = df_data_credit.id
    df_data_credit.sort_index(inplace=True)
    df_data_credit.drop(columns="id", inplace=True)

    # Close the cursor and connection
    cursor.close()
    conn.close()


except Exception as e:
    print(f"Error: {e}")


insert_data_sql = f"""
INSERT INTO credit_table ({", ".join(df_data_credit.columns)}) VALUES ({", ".join(["%s"] * len(df_data_credit.columns))});
"""

# inserting the shifted data
csv_file_path = "data/shifted_data.csv"
df = pd.read_csv(csv_file_path, index_col=[0])


import pandas as pd
import psycopg2  # Replace with the appropriate database library for your SQL database

try:
    conn = psycopg2.connect(
        host=db_host,
        database=db_name,
        user=db_user,
        password=db_password,
        port=5432,  # The port for PostgreSQL
    )
    cursor = conn.cursor()

    # Define the table name
    table_name = "credit_table"

    # Loop through the DataFrame and insert data row by row
    for index, row in df.iterrows():
        values = tuple(row)
        cursor.execute(insert_data_sql, values)

    # Commit the changes and close the cursor and connection
    conn.commit()
    cursor.close()
    conn.close()
except Exception as e:
    print(e)


# finally updating the current model verison
try:
    # Connect to the database
    conn = psycopg2.connect(
        host=db_host, database=db_name, user=db_user, password=db_password, port=5432
    )
    cursor = conn.cursor()
    print("Connected to the database!")

    cursor.execute(
        f"UPDATE credit_table SET model_version = {latest_versions[0].version}"
    )

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

except Exception as e:
    print(f"Error: {e}")
