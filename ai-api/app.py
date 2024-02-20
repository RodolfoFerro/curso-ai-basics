from datetime import datetime

from flask import Flask
from flask import request
from flask import jsonify
from flask import make_response
from joblib import load
import numpy as np

app = Flask(__name__)
model = load('assets/modelo_iris.joblib')
species = ['setosa', 'versicolor', 'virginica']


@app.route('/', methods=['GET'])
def base_route():
    headers = {'Content-Type': 'application/json'}
    message = {'message': 'Hello World!'}
    json_message = jsonify(message)

    return make_response(json_message, 200, headers)


@app.route('/fecha', methods=['GET'])
def today():
    headers = {'Content-Type': 'application/json'}
    message = {'today': datetime.now().strftime('%Y-%m-%d')}
    json_message = jsonify(message)

    return make_response(json_message, 200, headers)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    headers = {'Content-Type': 'application/json'}

    if request.method == 'GET':
        message = {'message': 'Hello World!'}
        json_message = jsonify(message)
        response = make_response(json_message, 200, headers)
        return response

    if request.method == 'POST':
        data = request.get_json()

        input_data = np.array([
            data['sepal_length'], data['sepal_width'], data['petal_length'],
            data['petal_width']
        ])

        prediction = model.predict([input_data])
        message = {
            'prediction': int(prediction[0]),
            'species': species[prediction[0]]
        }

        json_message = jsonify(message)
        response = make_response(json_message, 200, headers)
        return response


if __name__ == '__main__':
    app.run(port=5000, debug=True)
