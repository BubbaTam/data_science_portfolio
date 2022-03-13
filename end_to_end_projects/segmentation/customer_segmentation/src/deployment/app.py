import os
import numpy as np
import joblib
import pandas as pd
from src.config import MODEL_OUTPUT, PARAMETERS_OUTPUT
from flask import Flask, render_template,jsonify, request


# NEED TO Test
# - test when no inputs
# - changes to models / route to using different models


flask_app = Flask(__name__)

model = joblib.load(os.path.join(MODEL_OUTPUT,'kmeans_model.pkl'))
min_max_scaler = joblib.load(os.path.join(PARAMETERS_OUTPUT,'min_max_scaler.bin'))
gender_transformer = joblib.load(os.path.join(PARAMETERS_OUTPUT,'gender_transformer.bin'))


@flask_app.route("/")
@flask_app.route("/home")
def home():
    return render_template('home.html')



@flask_app.route("/predict",methods=["POST"])
def predict():
    ID_data = request.form["ID"]
    gender_input = request.form["Gender"]
    age_input = request.form["Age"]
    income_input = request.form["Income"]
    spending_input = request.form["Spending_score"]

    values_orig = [ID_data,gender_input,age_input,income_input,spending_input]
    values = [value for value in values_orig if value != '']

    # data preprocessing
    ## input data into a dataframe
    df = pd.DataFrame({'gender':[gender_input],'age':[age_input],'income':[income_input],'spending_score_(1-100)':[spending_input]})
    ## transform gender to integers from the original string
    df.iloc[:,0] = gender_transformer.transform(df.iloc[:,0])
    ## scale numerical features to between 0 and 1
    df.iloc[:,1:] = min_max_scaler.transform(df.iloc[:,1:])

    prediction = model.predict(df)
    return render_template("home.html",
                            inputs = values,
                            gender_input = gender_input,
                            age_input = age_input,
                            income_input = income_input,
                            spending_input = spending_input,
                            prediction = prediction)

@flask_app.route("/predict_api",methods=["POST","GET"])
def predict_api():
    """ to input a json data and to predict from the model"""
    if request.method == "GET":
        return render_template('home.html')
    if request.method == "POST":
        content = request.get_json()
        df = pd.read_json(content)
        ## transform gender to integers from the original string
        df.iloc[:,0] = gender_transformer.transform(df.iloc[:,0])
        ## scale numerical features to between 0 and 1
        df.iloc[:,1:] = min_max_scaler.transform(df.iloc[:,1:])
        prediction = model.predict(df)

        return str(prediction)