import pickle
#import pandas as pd
import numpy as np
from catboost import Pool
from flask import Flask, render_template, request
app = Flask(__name__)

with open('./data/catboost_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('./data/df.pkl', 'rb') as f:
    df = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    company = request.form['company']
    company_location = request.form['company_location']
    species = request.form['species']
    cocoa_p = float(request.form['cocoa_p'])
    REF = int(request.form['REF'])
    country = request.form['country']

    change = lambda x: "Yes" if x==1 else "No"
    criollo_bean = int(request.form['criollo_bean'])
    criollo_bean2 = change(criollo_bean)
    forastero_bean = int(request.form['forastero_bean'])
    forastero_bean2 = change(forastero_bean)
    is_blend = int(request.form['is_blend'])
    is_blend2 = change(is_blend)

    param_list = [company, company_location, species, cocoa_p, REF,
    country, criollo_bean2, forastero_bean2, is_blend2]

    param = pd.DataFrame([{'company':company,'company_location':company_location,
    'species':species,'cocoa_p':cocoa_p,'REF':REF,'country':country,
    'criollo_bean':criollo_bean,'forastero_bean':forastero_bean,'is_blend':is_blend}])
    global df
    df_len = df.shape[0]
    df = pd.concat([df, param]).reset_index(drop=True)

    l = ['REF', 'cocoa_p']
    for name in l:
        mean = np.nanmean(df[name], axis=0)
        std = np.nanstd(df[name], axis=0)
        df[name] = (df[name] - mean)/std

    X = df[['company','company_location','species','cocoa_p','REF',
        'country','criollo_bean','forastero_bean','is_blend']]
    y = df['rating']
    x_test = X[df_len:]
    y_test = y[df_len:]

    test_pool = Pool(x_test.values, y_test.values, cat_features=[0, 1, 2, 5, 6, 7, 8])
    y_pred = model.predict(test_pool)

    return render_template('result.html', param_list=param_list, y_pred=y_pred)

if __name__ == "__main__":
    app.run(debug=True, port=4040)
