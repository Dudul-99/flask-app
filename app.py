from flask import Flask, render_template, request
import joblib 
import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn import set_config

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])


def main():
    
    # If a form is submitted
    if request.method == "POST":
        
        #  classifier
        best_model_loaded = joblib.load("model.joblib")
        # Import the pipeline transformer
        full_pipeline_loaded = joblib.load("full_pipeline.joblib")
        # Get values through input bars
        RoomService = request.form.get("RoomService")
        FoodCourt = request.form.get("FoodCourt")
        ShoppingMall = request.form.get("ShoppingMall")
        Spa = request.form.get("Spa")
        VRDeck = request.form.get("VRDeck")
        Age = request.form.get("Age")
        luxury = request.form.get("luxury")
        is_travel_group = request.form.get("is_travel_group")
        family_journey = request.form.get("family_journey")
        CryoSleep = request.form.get("CryoSleep")
        VIP = request.form.get("VIP")
        HomePlanet = request.form.get("HomePlanet")
        Destination = request.form.get("Destination")
        deck = request.form.get("deck")
        side= request.form.get("side")
        
        # Put inputs to dataframe
        conver_dict = {
        "RoomService": float,
        "FoodCourt": float,
        "ShoppingMall": float,
        "Spa": float,
        "VRDeck": float,
        "Age": float,
        "luxury": int,
        "is_travel_group": object,
        "family_journey": object,
        "CryoSleep": object,
        "VIP": object,
        "HomePlanet": object,
        "Destination": object,
        "deck": object,
        "side": object
        }

        X = pd.DataFrame([[HomePlanet, CryoSleep,Destination,Age,VIP,RoomService,FoodCourt,
                           ShoppingMall,Spa,VRDeck,is_travel_group,family_journey,deck,
                           side,luxury]], columns=["HomePlanet", "CryoSleep","Destination","Age","VIP",
                                                   "RoomService","FoodCourt","ShoppingMall","Spa","VRDeck",
                                                  "is_travel_group","family_journey","deck","side","luxury"])
        # conversion
        X = X.astype(conver_dict)
        # pipline transformation
        set_config(transform_output="pandas")
        processesed = full_pipeline_loaded.fit_transform(X)
        # Get feature names from the model
        feature_names = best_model_loaded.get_booster().feature_names   
        # Create DataFrame with features as columns
        data = pd.DataFrame(columns=feature_names, data=X)
        # fill with NA for value not present
        data.fillna(0, inplace=True)
        
        print(processesed.columns)

        # Get prediction
        prediction =best_model_loaded.predict(data)[0]
        if prediction == 1:
            prediction = "You have been teleported during the trip !"
        else:
            prediction = "Congratulation, you have not been teleported during the trip !"
    else:
        prediction = ""

   
    return render_template("frontend.html", output = prediction)

if __name__ == '__main__':
    app.run(debug = True)
