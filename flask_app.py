import pickle
from flask import Flask, request, jsonify, render_template
import warnings


app = Flask(__name__)

# Load your machine learning model
with open('cluster.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
with open('scaler.pkl', 'rb') as scale:
    scaler= pickle.load(scale)
warnings.filterwarnings('ignore')
results=['High credit,long duration,Young customer','Low credit,Short duration,Young customer','Low credit,Short duration,Old customer','High credit,Mid Duration,Old Customer']


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
        # try:
        data = request.get_json()
        print(data)
        age = data['age']
        credit_amount = data['credit_amount']
        duration = data['duration']

        # Perform preprocessing (if needed) on the input data
        # You might need to convert inputs to the appropriate format for your model

        # Make a prediction using the loaded machine learning model
        prediction = model.predict(scaler.transform([[age, credit_amount, duration]]))
        # You can return the prediction as JSON
        print(prediction)
        result = {'prediction': results[prediction[0]]}
        return jsonify(result)
    # except Exception as e:
    #     print(str(e))
    #     return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
