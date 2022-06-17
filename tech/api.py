from email import message
import string
from sys import api_version
from typing import Text
from numpy import append, float64, int64, integer, product
import pandas as pd
import pickle
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from requests import request

# API instantiation

api = FastAPI()

if __name__ == '__main__':
    uvicorn.run(api, host= '127.0.0.1', port = 5001)

# Load model from the serialized pkl file
pickle_filename = '../churnmodel.pkl'
with open(pickle_filename, 'rb') as file:
    clf = pickle.load(file)

# model for data validation
class Churn(BaseModel):
    # Row:integer
    # Id: integer
    # Surname: string
    Score: int64
    Nationality: object
    Gender: object
    Age: int64
    Tenure: int64
    Balance: float64
    Product: int64
    Card: int64
    Active: int64
    Salary: float64
    Exited: int64



# Defining root path and message
@api.get('/')
def root():
    return {message: 'Hello fraands!'}

# define a pred. endpoint

# Defining the prediction endpoint with data validation
@api.post('/predict')
async def predict(churn: Churn):
	
	# Converting input data into Pandas DataFrame
	input_df = pd.DataFrame([churn.dict()])
	
	# Getting the prediction from the Logistic Regression model
	pred = clf.predict(input_df)[0]
	return pred

