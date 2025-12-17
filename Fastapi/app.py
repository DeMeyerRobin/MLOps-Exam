from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import json

app = FastAPI(title="GOT House Predictor API")

# Load model and feature columns
model = joblib.load("model_output/model.pkl")

with open("Fastapi/feature_columns.json", "r") as f:
    feature_cols = json.load(f)["columns"]

class CharacterInput(BaseModel):
    region: str
    primary_role: str
    alignment: str
    status: str
    species: str
    honour_1to5: int
    ruthlessness_1to5: int
    intelligence_1to5: int
    combat_skill_1to5: int
    diplomacy_1to5: int
    leadership_1to5: int
    trait_loyal: bool
    trait_scheming: bool

@app.get("/")
def home():
    return {
        "message": "GOT House Predictor API",
        "usage": "POST to /predict with character JSON",
        "example": {
            "region": "The North",
            "primary_role": "Commander",
            "alignment": "Lawful Good",
            "status": "Alive",
            "species": "Human",
            "honour_1to5": 4,
            "ruthlessness_1to5": 2,
            "intelligence_1to5": 3,
            "combat_skill_1to5": 4,
            "diplomacy_1to5": 3,
            "leadership_1to5": 4,
            "trait_loyal": True,
            "trait_scheming": False
        }
    }

@app.post("/predict")
def predict(character: CharacterInput):
    # Convert to dataframe
    input_dict = character.dict()
    df = pd.DataFrame([input_dict])
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, dummy_na=True)
    
    # Create empty dataframe with training features
    input_final = pd.DataFrame(columns=feature_cols)
    
    # Fill in the features we have
    for col in df_encoded.columns:
        if col in feature_cols:
            input_final[col] = df_encoded[col].values
    
    # Fill missing features with 0
    input_final = input_final.fillna(0)
    
    # Predict
    prediction = model.predict(input_final)[0]
    
    return {
        "predicted_house": prediction,
        "input": input_dict
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)