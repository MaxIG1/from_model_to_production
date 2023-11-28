import datetime as dt
from flask import Flask, render_template
from flask import request
from flask import jsonify
import mlflow
import mlflow.sklearn
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from scipy import stats
import statsmodels.api as sm
import numpy as np
from flask_httpauth import HTTPBasicAuth
from apscheduler.schedulers.background import BackgroundScheduler
from threading import Thread
from dotenv import load_dotenv
import os


# for conveniennce, creating a experiment and a model name
number = 51
experiment_name = f"experiment_{number}"
model_name = f"model_{number}"


# creating the class that will handle the data
class DataRetriever:
    def __init__(self, db_host, db_name, db_user, db_password):
        self.db_host = db_host
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.model_registered_name = model_mangement.model_registered_name

    # basic function that gets all the available data from the postgress sql database
    def get_data(self):
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=5432,
            )

            cursor = conn.cursor()
            print("Connected to the database for data retrieval!")

            cursor.execute("SELECT * FROM credit_table")
            rows = cursor.fetchall()

            column_list = [column[0] for column in cursor.description]
            df_data_credit = pd.DataFrame(rows, columns=column_list)
            df_data_credit = df_data_credit.set_index("id")

            cursor.close()
            conn.close()
            return df_data_credit, column_list

        except Exception as e:
            print(e)

    # basic function that gets the users and their passwords for authentication into the Flask API
    def password_db_connector(self, username, password):
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=5432,
            )

            cursor = conn.cursor()
            print("Connected to the database for password!")

            cursor.execute("SELECT * FROM user_table")
            rows = cursor.fetchall()

            column_list = [column[0] for column in cursor.description]
            user_df = pd.DataFrame(rows, columns=column_list)
            user_df.index = user_df.id
            user_df.sort_index(inplace=True)
            user_df.drop(columns="id", inplace=True)

            if username in user_df["username"].values:
                cursor.close()
                conn.close()
                return (
                    user_df.loc[user_df["username"] == username, "password"].values[0]
                    == password
                )

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error: {e}")

        return False

    # this function will insert the values, a user wants a prediction for into the sql database and will flag it as
    # new data. It takes the data_dict from the prediction function in the flask section.
    def insert_sql(self, data_dict):
        try:
            conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password,
                port=5432,
            )

            cursor = conn.cursor()
            print("Connected to the database for inserting!")
            # this statement will create wild cards for the sql insert.
            insert_sql = """
            INSERT INTO credit_table (Creditability, Account_Balance, Duration_of_Credit_month, Payment_Status_of_Previous_Credit, Purpose, Credit_Amount, Value_Savings_Stocks, Length_of_current_employment, Instalment_per_cent, Sex_Marital_Status, Guarantors, Duration_in_Current_address, Most_valuable_available_asset, Age_years, Concurrent_Credits, Type_of_apartment, No_of_Credits_at_this_Bank, Occupation, No_of_dependents, Telephone, Foreign_Worker, new_column, model_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
            """

            # the data comes in a packed in tuples, which is why they need to be unpacked.
            unpacked_data_dict = {key: value[0] for key, value in data_dict.items()}
            values_list = list(unpacked_data_dict.values())

            values_list.insert(0, int(1))
            # appending the true value, which implies that the data is new
            values_list.append(True)

            # appedning the current version of the model used
            latest_versions = client.get_latest_versions(
                name=self.model_registered_name
            )
            values_list.append(latest_versions[0].version)

            # transforming to tuples because sql needs that
            values_tuple = tuple(values_list)

            cursor.execute(insert_sql, values_tuple)
            conn.commit()

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error: {e}")

    # this function is used after the retraining has been used, it will update the information if the data has been used
    # for retraining
    def updating_new_column(self):
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
            cursor.execute(
                "UPDATE credit_table SET new_column = False WHERE new_column = True"
            )
            conn.commit()

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error: {e}")

    # this is function is also used after retraining. It will update the verion info on the new data.
    def updating_version_info(self):
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

            latest_versions = client.get_latest_versions(
                name=self.model_registered_name
            )

            # Select all values from the table
            cursor.execute(
                f"UPDATE credit_table SET model_version = {latest_versions[0].version} WHERE new_column = True"
            )
            conn.commit()

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error: {e}")


# run the retrainining once per month
def background_task():
    scheduler = BackgroundScheduler()
    # Schedule the retrain function to run once per month on the first day at 13:00 o'clock.

    scheduler.add_job(
        lambda: model_mangement.retrain(data_retriever.get_data()[0], 0),
        "cron",
        day="1",
        hour="13",
        minute="00",
    )
    scheduler.start()


# this class handels the model management
class Model_Management:
    def __init__(self, registered_model_name):
        self.model_registered_name = registered_model_name
        self.model_version = "latest"

    # the function that loads the most current registered model
    def load_model(self):
        model = mlflow.pyfunc.load_model(
            model_uri=f"models:/{self.model_registered_name}/latest"
        )
        return model

    ##(re-)training the model
    # initial 0 means retraining, not initial
    # function 1 means retraining because initial.
    def retrain(self, data, initial):
        if initial == 0:  # basic retrain, then only the new data, else all data
            data = data[data["new_column"] == True]
        elif initial == 1:
            # it will take all available old data.
            data = data[data["new_column"] == False]
        X = data.drop(columns=["creditability", "new_column", "model_version"])
        y = data["creditability"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create a Random Forest classifier
        rf_classifier = RandomForestClassifier()

        # Defineing a parameter grid and the GridSeach object
        # hence I upload this on a slow server, just the basics, works ok.
        param_grid = {
            "n_estimators": [10, 100, 200, 300],
            "max_depth": [None, 10, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True],
        }

        # initiating the grid_search object
        grid_search = GridSearchCV(
            estimator=rf_classifier,
            param_grid=param_grid,
            scoring="accuracy",
            cv=5,
            n_jobs=-1,
            verbose=3,
        )

        print("training about to start")

        # Fiting
        grid_search.fit(X, y)

        # Accessing the best hyperparameters and the best classifier from the grid search
        best_params = grid_search.best_params_
        model = grid_search.best_estimator_

        # Evaluate the best classifier on your test set and generate a classification report
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        # these prints like others give some information for trouble shooting in the server console.
        print(report)

        if int(initial) == 0:  # that is basic retrain
            # just to be sure.
            mlflow.end_run()

            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():  # as run
                # Log the model to MLflow
                client = mlflow.tracking.MlflowClient()

                mlflow.sklearn.log_model(
                    sk_model=model,
                    # this is important because if there is a path, it will double the path, which makes it crash
                    # not sure if this is a bug on mlflow side
                    artifact_path="",
                    registered_model_name=self.model_registered_name,
                )

                # Log the classification report and accuracy
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_param("best_params", best_params)
                mlflow.log_text(report, "classification_report.txt")

        # the initial logging of the model, which will register the model. Necessary if training start on the data set
        elif int(initial) == 1:  # initial
            mlflow.end_run()
            mlflow.set_experiment(experiment_name)

            with mlflow.start_run():  # as run
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=f"runs:/{mlflow.active_run().info.run_id}/{model_name}",
                    registered_model_name=self.model_registered_name,
                )

                # Log the classification report and accuracy and the params
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_param("best_params", best_params)
                mlflow.log_text(report, "classification_report.txt")
                #

            # setting the model version in the data database
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
                client = mlflow.tracking.MlflowClient()
                latest_versions = client.get_latest_versions(
                    name=self.model_registered_name
                )

                # Select all values from the table
                cursor.execute(
                    f"UPDATE credit_table SET model_version = {latest_versions[0].version}"
                )
                conn.commit()

                cursor.close()
                conn.close()

            except Exception as e:
                print(f"Error: {e}")

    # The next function will perform the testings with the adequat bonferoni correction
    # Four tests are choosen, t-welch for numerical data, z-test for proportions for bernouli data
    # furthermore chi-square test for nomianal and mannwhitneyu for ordinal data.
    def testing(self, data):
        # creating a list of the scale types of the data, so that the right test can be chosen.
        qual_list = []

        for col in data:
            unique_values = data[col].unique()
            if len(unique_values) > 10:
                qual_list.append("num")
            elif len(unique_values) <= 10 and len(unique_values) > 2:
                qual_list.append("qual")
            else:
                qual_list.append("bernoulli")

        # the qualitative data needs to be distinguished between ordinal and nominal by hand.
        # Furthermore some are numerical even if they dont seem like it.
        qual_list[8] = "num"
        qual_list[16] = "num"
        qual_list[1] = "nom"
        qual_list[3] = "nom"
        qual_list[4] = "nom"
        qual_list[6] = "nom"
        qual_list[7] = "ord"
        qual_list[9] = "nom"
        qual_list[10] = "nom"
        qual_list[11] = "ord"
        qual_list[12] = "nom"
        qual_list[14] = "nom"
        qual_list[15] = "nom"
        qual_list[17] = "ord"
        qual_list = qual_list[:-2]

        latest_versions = client.get_latest_versions(name=self.model_registered_name)

        print(f"the latest version is {latest_versions[0].version}")
        # creating two dataframes with old and with new values, yet we only use
        # the data of the current version, because data drift here has been accounted for already
        old_data = data[
            (data["new_column"] != True)
            & (data["model_version"] == int(latest_versions[0].version))
        ]
        print(old_data)
        new_data = data[
            (data["new_column"] == True)
            & (data["model_version"] == int(latest_versions[0].version))
        ]
        print(new_data)
        test_results = {}

        for column, qual in zip(data.columns[1:-1], qual_list[1:-1]):
            # calculating the t-welch test for numerical variables:
            if qual == "num":
                t_stat, p_value = stats.ttest_ind(
                    old_data[column], new_data[column], equal_var=False
                )
                test_results[column] = {
                    "t_stat": t_stat,
                    "p_value": p_value,
                    "type": qual,
                }

            # calculating the z proportional test for bernouli variables
            elif qual == "bernoulli":
                print(column)
                success_group_1 = old_data[column].sum()
                total_group_1 = len(old_data[column])
                success_group_2 = new_data[column].sum()
                total_group_2 = len(new_data[column])
                z_stat, p_value = sm.stats.proportions_ztest(
                    [success_group_1, success_group_2], [total_group_1, total_group_2]
                )
                test_results[column] = {
                    "z_stat": z_stat,
                    "p_value": p_value,
                    "type": qual,
                }

            # calculating the chi square test for nominal variables
            elif qual == "nom":
                # Calculate the old frequencies for the 'old_data' dataset
                old_frequencies = old_data[column].value_counts()
                # Calculate the new frequencies based on the 'new_data' dataset
                new_frequencies = new_data[column].value_counts()

                # create a set of total indexes and fill missing with 0
                all_values = set(old_frequencies.index) | set(new_frequencies.index)
                old_frequencies = old_frequencies.reindex(all_values, fill_value=0)
                new_frequencies = new_frequencies.reindex(all_values, fill_value=0)

                # chi square lets go baby
                contingency_table = pd.concat(
                    [old_frequencies, new_frequencies], axis=1
                )
                contingency_table.columns = ["Old Data", "New Data"]
                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                test_results[column] = {"chi2": chi2, "p_value": p_value, "type": qual}

            # Perform the Wilcoxon rank-sum test for ordinal data
            elif qual == "ord":
                statistic, p_value = stats.mannwhitneyu(
                    old_data[column], new_data[column]
                )
                test_results[column] = {
                    "Mannwhitneyu": statistic,
                    "p_value": p_value,
                    "type": qual,
                }
        # if any of the p-values is smaller than the p value it return true according to the bonferoni correction
        for column in test_results:
            if test_results[column]["p_value"] <= (1 - np.e ** (np.log(0.95) / 20)):
                return True


# from here on, the flask endpoints are configured
# furthermore the flask class and the authenitification class is initiated
app = Flask(__name__)
auth = HTTPBasicAuth()


# the password verifying function that connects to the user and password database
@auth.verify_password
def verify(username, password):
    return data_retriever.password_db_connector(username, password)


# a test function, to see if flask is running
@app.route("/hello")
@auth.login_required
def hello_world():
    return "hello"


# if manual initial training is wished for.
@app.route("/initialtrain")
@auth.login_required
def init_train_handle():
    data = data_retriever.get_data()[0]
    model_mangement.retrain(data, 1)
    return "Training successful"


# if manual retraining is wished for.
@app.route("/retrain")
@auth.login_required
def retrain_handle():
    data = data_retriever.get_data()[0]
    model_mangement.retrain(data, 0)
    return "Retraining successful"


# the predict function.
# it will predict and also test if data drift has happened via the testing function.
@app.route("/predict", methods=["GET", "POST"])
@auth.login_required
def predict():
    data_dict = {}
    # here the get method
    if request.method == "GET":
        # loading the data and the name of the columns
        column_list = data_retriever.get_data()[1]

        # Loop through query parameters according to the collumn list
        # and creating the data_dict from it.
        for i, column_name in zip(range(1, 21), column_list[2:-2]):
            param_value = request.args.get(column_name)

            if param_value is not None:
                try:
                    num = float(param_value)
                    data_dict[column_name] = [num]
                except ValueError:
                    return (
                        jsonify(
                            {
                                "error": f"Invalid input for {column_name}. Please provide a valid number."
                            }
                        ),
                        400,
                    )

    # here the post method
    elif request.method == "POST":
        # Handle POST request
        try:
            data_dict = request.get_json()
            data_dict = {key: [float(value)] for key, value in data_dict.items()}
        except Exception as e:
            return jsonify({"error": f"Invalid JSON payload. {str(e)}"}), 400

    print(data_dict)
    # inserting the user input into the sql database
    data_retriever.insert_sql(data_dict)

    # loading the data after inserting
    df_data_credit = data_retriever.get_data()[0]

    # load the model with the following logic if a retraining has happend, load the new model
    # also load if now model has been loaded (e.g. initial loading)
    global loaded_model

    # after 1000 entries checking if datadrift is existent
    if df_data_credit.new_column.sum() >= 1000:
        testing_check = model_mangement.testing(df_data_credit)

        # if true retraining, reloading the new model, updating the column version in the database which formed the basis
        # and finally reseting the value of the data from new to old.
        if testing_check == True:
            print("retrain")
            model_mangement.retrain(data_retriever.get_data()[0], 0)
            print("retraining is over")
            loaded_model = model_mangement.load_model()
            # updating the model version, in the database according if it is the basis of the model
            # on important thing here. If there is a new model , the first thousend entries where predicted with the prior version, but
            # hence datadrift was detected, the thousand entries build the basis for the new model. So the first thousand entries of a new model
            # build the basis for the new model, yet were used for prediction with the prior model version.
            data_retriever.updating_version_info()
            # updating the information if the data has been accounted for (changing the values from new to old)
            data_retriever.updating_new_column()
        else:
            # if testing false then updating and integrating the data in the seminal data corpus
            data_retriever.updating_new_column()
    else:
        try:
            # if no model is loaded it obviously creates a NameError
            if loaded_model:
                pass
        except NameError:
            loaded_model = model_mangement.load_model()

    # Prediction
    # Create a DataFrame from the data dictionary for prediction
    if request.method == "GET":
        df_predict = pd.DataFrame(data_dict)
    elif request.method == "POST":
        df_predict = pd.DataFrame(data_dict, index=[0])

    # logging the experiment
    # Measure the time before making predictions
    start_time = dt.datetime.now()
    # Now lets make predictions using the loaded model
    predictions = loaded_model.predict(df_predict)

    # Measure the time after making predictions
    end_time = dt.datetime.now()
    prediction_info = {
        "input_values": df_predict.iloc[0].tolist(),
        "prediction_time": str(end_time - start_time),
        "predictions": predictions.tolist(),
    }

    # just to be sure.
    if mlflow.active_run():
        mlflow.end_run()

    # Use mlflow.set_experiment to set the active experiment for storing predictions
    experiment = mlflow.set_experiment("stored_predictions")
    with mlflow.start_run(experiment_id=experiment.experiment_id):
        # Log the input values, prediction time, and predictions
        mlflow.log_params(prediction_info)
        mlflow.end_run()

    # the sentence
    if predictions[0] == 0:
        sentence = f"The person might intend to commit fraud."
    else:
        sentence = f"The person is probably not intending to commit fraud."

    response_data = {"message": sentence}
    http_status_code = 200

    # webbrowser answer
    if request.method == "GET":
        # the html response containing the sentence
        return render_template(
            "index.html", response_data=response_data, http_status_code=http_status_code
        )
    # restful api confornm
    elif request.method == "POST":
        return jsonify(response_data), http_status_code


if __name__ == "__main__":
    # Start the background task in a separate thread
    background_thread = Thread(target=background_task)
    background_thread.daemon = True
    background_thread.start()

    # loading passwords and the like
    load_dotenv(dotenv_path="credentials.env")

    db_host = os.getenv("DB_HOST")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")

    # initaite the classes
    model_mangement = Model_Management(f"register_{number}")
    data_retriever = DataRetriever(db_host, db_name, db_user, db_password)

    # passwords and username for ml_flow server
    MLFLOW_TRACKING_USERNAME = os.getenv("MLFLOW_TRACKING_USERNAME")
    MLFLOW_TRACKING_PASSWORD = os.getenv("MLFLOW_TRACKING_PASSWORD")

    # setting the tracking server
    TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")

    # Construct the tracking URI with credentials
    tracking_uri_with_credentials = f"http://{MLFLOW_TRACKING_USERNAME}:{MLFLOW_TRACKING_PASSWORD}@{TRACKING_SERVER_HOST}:5000"

    # Set the tracking URI
    mlflow.set_tracking_uri(tracking_uri_with_credentials)

    # create the client that is need to return the current model version
    client = mlflow.tracking.MlflowClient()
    app.run(host="0.0.0.0", port=5001, debug=True)
