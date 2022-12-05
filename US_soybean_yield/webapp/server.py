from flask import Flask, jsonify, request, render_template
import pandas as pd
import joblib

app = Flask(__name__)

feature_order = ["lat",\
                 "lon",\
                 "year",\
                 "JJA_SWdown",\
                 "JJA_LWnet",\
                 "JFMA_EF",\
                 "JJA_EF",\
                 "SO_EF",\
                 "JFMA_Rainf",\
                 "MJJ_Rainf",\
                 "JF_Snowf",\
                 "JJA_ESoil",\
                 "MJJA_Albedo",\
                 "MJJA_SoilM_0_10cm",\
                 "JFMA_RootMoist",\
                 "JanToApr_LAI",\
                 "MayToOct_ACond",\
                 "Lead_AnnualTotal_Rainf",\
                 "ECanop_Jan",\
                 "ACond_Jan",\
                 "Qle_Jan",\
                 "GVEG_Apr",\
                 "SoilM_100_200cm_Sep",\
                 "LWnet_May",\
                 "ESoil_May",\
                 "AvgSurfT_Aug"]

@app.route("/predict", methods=['POST'])
def yield_prediction():
    # ------------------------------------------#
    # solicit input and turn input to dataframe #
    # ------------------------------------------#
    json = request.get_json()
    df   = pd.DataFrame(json, index=[0])

    # ----------------- #
    # load the model    #
    # ------------------#
    model= joblib.load('model/xgboost_model.pkl')

    # -------------------------------------------------#
    # make sure the columns follow  the feature_order  #
    # -------------------------------------------------#
    df = df[feature_order]

    # ----------- #
    #   predict   #
    # ------------#
    y_predict   = model.predict(df.values)

    # ----------------- #
    # return the result #
    # ----------------- #
    result      = {"Predicted_Yield": float(y_predict[0])}
    return jsonify(result)

@app.route("/")
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0')
