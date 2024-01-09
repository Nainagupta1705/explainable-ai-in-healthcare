import os
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


def predict(values, dic):

    # breast_cancer
    if len(values) == 30:
        model = pickle.load(open('breast-model.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]

    # heart disease
    elif len(values) == 13:
        model = pickle.load(open('heart-model.pkl', 'rb'))
        values = np.asarray(values)
        return model.predict(values.reshape(1, -1))[0]


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/cancer", methods=['GET', 'POST'])
def cancerPage():
    return render_template('breast_cancer.html')


@app.route("/heart", methods=['GET', 'POST'])
def heartPage():
    return render_template('heart.html')


@app.route("/breastcancer_i", methods=['GET', 'POST'])
def breastcancer_iPage():
    return render_template('breastcancer_i.html')


@app.route("/predict", methods=['POST', 'GET'])
def predictPage():
    # try:
    if request.method == 'POST':
        to_predict_dict = request.form.to_dict()

        for key, value in to_predict_dict.items():
            try:
                to_predict_dict[key] = int(value)
            except ValueError:
                to_predict_dict[key] = float(value)

        to_predict_list = list(map(float, list(to_predict_dict.values())))
        pred = predict(to_predict_list, to_predict_dict)

    return render_template('predict.html', pred=pred)


@app.route("/breastcancer_ipredict", methods=['POST', 'GET'])
def breastcancer_ipredictPage():
    if request.method == 'POST':

        return render_template('404.html')


if __name__ == '__main__':
    app.run(debug=True)
