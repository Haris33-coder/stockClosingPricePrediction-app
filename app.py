import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model_VR = pickle.load(open('./models/model_VR.pkl', 'rb'))
model_RFR = pickle.load(open('./models/model_RFR.pkl', 'rb'))
model_BRR = pickle.load(open('./models/model_BRR.pkl', 'rb'))
model_DTR = pickle.load(open('./models/model_DTR.pkl', 'rb'))
model_GBR = pickle.load(open('./models/model_GBR.pkl', 'rb'))
model_KRR = pickle.load(open('./models/model_KRR.pkl', 'rb'))
model_LR = pickle.load(open('./models/model_LR.pkl', 'rb'))
model_LSS = pickle.load(open('./models/model_LSS.pkl', 'rb'))
model_MLP = pickle.load(open('./models/model_MLP.pkl', 'rb'))
model_SVR = pickle.load(open('./models/model_SVR.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # request.form['open']
    vr = model_VR.predict(final_features)
    rfr = model_RFR.predict(final_features)
    brr = model_BRR.predict(final_features)
    dtr = model_DTR.predict(final_features)
    gbr = model_GBR.predict(final_features)
    krr = model_KRR.predict(final_features)
    lr = model_LR.predict(final_features)
    lss = model_LSS.predict(final_features)
    mlp = model_MLP.predict(final_features)
    svr = model_SVR.predict(final_features)

    svr = round((svr[0]), 2)
    vr = round((vr[0]), 2)
    rfr = round((rfr[0]), 2)
    brr = round((brr[0]), 2)
    dtr = round((dtr[0]), 2)
    gbr = round((gbr[0]), 2)
    krr = round((krr[0]), 2)
    lr = round((lr[0]), 2)
    mlp = round((mlp[0]), 2)
    lss = round((lss[0]), 2)

    prediction = [["Support Vector Regsessor", svr], ["Linear Regressor", lr],
                  ["Random Forest Regressor", rfr], ["Bayesian Ridge Regressor", brr],
                  ["Kernal Ridge Regressor", krr], ["Decision Tree Regressor", dtr],
                  ["Gradient Boosting Regressor", gbr], ["Laso Regressor", lss],
                  ["Multi-Layered Perceptron", mlp], ["Voting Regressor", vr]]

    return render_template('index.html', prediction = prediction)
    
if __name__ == "__main__":
    app.run(debug=True)