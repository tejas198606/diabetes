import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask(__name__,template_folder='template')
model = pickle.load(open('model003.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('diab.html')   


@app.route('/predict',methods=['POST'])
def predict():                               
    input_features = [float(x) for x in request.form.values()]
    final_features = [np.array(input_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0], 2)
    
    features_name = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']
    
    
    df = pd.DataFrame(final_features, columns=features_name)
    output = model.predict(df)
    
    return render_template('diab.html', prediction_text=' Outcome {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)