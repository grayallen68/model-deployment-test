import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model2.pkl', 'rb'))

outputs = [
    'Death awaits you',
    'You get to survive'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    #convert entered features to desired datatypes
    # int_features = [x.toint() for x in request.form.values()]
    # print((request.form.values))
    # int_features = [int(x) for x in request.form.values()]
    features = request.form.to_dict()
    features = [features['Pclass'], features['Age'], features['Sex']]
    int_features = [int(f) for f in features]
    
    print('-----INIT-----{}'.format(int_features))

    final_features = [np.array(int_features)]
    print('---FINAL--- {}'.format(final_features))
    prediction = model.predict(final_features)
    print('---PREDICTION---- {}'.format(prediction))
    output = outputs[prediction[0]]

    return render_template('index.html', prediction_text='{} on the titanic'.format(output))


if __name__ == "__main__":
    app.run(debug=True)