# %%
import requests
from requests.auth import HTTPBasicAuth

url = "http://127.0.0.1:5001/predict"

# Example data in dictionary format
data_dict = {
    "account_balance": 4,
    "duration_of_credit_month": 12,
    "payment_status_of_previous_credit": 4,
    "purpose": 3,
    "credit_amount": 618,
    "value_savings_stocks": 1,
    "length_of_current_employment": 5,
    "instalment_per_cent": 4,
    "sex_marital_status": 3,
    "guarantors": 1,
    "duration_in_current_address": 4,
    "most_valuable_available_asset": 2,
    "age_years": 56,
    "concurrent_credits": 3,
    "type_of_apartment": 2,
    "no_of_credits_at_this_bank": 1,
    "occupation": 1,
    "no_of_dependents": 3,
    "telephone": 0,
    "foreign_worker": 0,
}


# Authentication credentials
username = "merlin"
password = "youshallnotpass"

# Send a POST request with JSON payload and authentication
response = requests.post(url, json=data_dict, auth=HTTPBasicAuth(username, password))

# Print the response
print(response.status_code)
print(response.json())
