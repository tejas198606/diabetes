import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Pregnancies':2.0000,'Glucose':9.0000,'BloodPressure':6.0000,'SkinThickness':2.00000,'Insulin':9.00000,'BMI':6.00000,
                            'DiabetesPedigreeFunction':9.00000,'Age':6.00000})

print(r.json())