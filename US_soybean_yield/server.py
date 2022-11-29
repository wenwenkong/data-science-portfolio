from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def yield_prediction():
    json = request.get_json()
    model= joblib.load('model/xgboost_model.pkl')
    df   = pd.DataFrame(json, index=[0])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    
    df_x_scaled = scaler.fit_transform(df)
    df_x_scaled = pd.DataFrame(df_x_scaled, columns=df.columns)
    y_predict   = model.predict(df_x_scaled)

    result      = {"Predicted soybean annual yield in 2016": float(y_predict[0])}
    return jsonify(result)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0')
