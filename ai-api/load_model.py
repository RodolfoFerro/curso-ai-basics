from joblib import load


model = load('assets/modelo_iris.joblib')

prediction = model.predict([[5.2, 2.7, 3.9, 1.4]])
print(prediction)
