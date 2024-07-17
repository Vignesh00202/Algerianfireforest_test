import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application


## import ridge regressor and scaler pickle
ridge_model=pickle.load(open('models/ridge.pkl','rb'))
standardscaler=pickle.load(open('models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/prediction",methods=['GET','POST'])
def predict_datapoint():
   if request.method=="POST":
    temperature=float(request.form.get('temperature'))
    rh=float(request.form.get('rh'))
    ws=float(request.form.get('ws'))
    rain=float(request.form.get('rain'))
    ffmc=float(request.form.get('ffmc'))
    dmc=float(request.form.get('dmc'))
    isi=float(request.form.get('isi'))
    classes=float(request.form.get('classes'))
    region=float(request.form.get('region'))

    new_data_scaled=standardscaler.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
    result=ridge_model.predict(new_data_scaled)

    return render_template('home.html',results=result[0])
   else:
    return render_template("home.html")

if __name__=="__main__":
    app.run(host="0.0.0.0")
