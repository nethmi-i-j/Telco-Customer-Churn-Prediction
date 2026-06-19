import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ModelInput(BaseModel):
    gender:           str
    SeniorCitizen:    str      
    Partner:          str
    Dependents:       str
    tenure:           int
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str
    MonthlyCharges:   float
    TotalCharges:     float

churn_model = joblib.load('Telco_customer_churn_xgboost_model.joblib')

@app.post('/customer_churn_prediction')
def churn_pred(input_parameters: ModelInput):
    # DataFrame with correct column names, in training order
    input_df = pd.DataFrame([input_parameters.model_dump()])

    prediction = churn_model.predict(input_df)

    return {
        'label': int(prediction[0]),
        'result': 'Churn' if prediction[0] == 1 else 'Non Churn'
    }