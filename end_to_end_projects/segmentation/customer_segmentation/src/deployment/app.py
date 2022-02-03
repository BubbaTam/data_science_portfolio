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
    values = [ID_data,gender_input,age_input,income_input,spending_input]


    # data preprocessing
    df = pd.DataFrame({'gender':[gender_input],'age':[age_input],'annual_income_(k$)':[income_input],'spending_score_(1-100)':[spending_input]})
    gender_mapping = {'Female':0,'Male':1}
    df.gender = df.gender.replace(gender_mapping)

    scaler = joblib.load(os.path.join(PARAMETERS_OUTPUT,'min_max_scaler.bin'))
    df.iloc[:,1:] = scaler.transform(df.iloc[:,1:])

    prediction = model.predict(df)
    return render_template("home.html",
                            inputs = values,
                            gender_input = gender_input,
                            age_input = age_input,
                            income_input = income_input,
                            spending_input = spending_input,
                            prediction = prediction)

@flask_app.route("/predict_API",methods=["POST"])
def predict_API():
    """ to input a json data and to predict from the model"""
    pass

flask_app.run(debug=True)