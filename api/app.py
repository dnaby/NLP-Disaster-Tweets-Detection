# Ajouter le répertoire parent pour les imports de module
import sys
sys.path.append('..')

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pickle
import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.metrics import classification_report, confusion_matrix
import dagshub
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.models.main import (
  read_and_split_data
)

# Initialize DagsHub and MLflow
dagshub.init(repo_owner='dnaby', repo_name='NLP-Disaster-Tweets-Detection', mlflow=True)

logged_model = 'runs:/f0d16e0cafa54483830c0d104cb3d58a/model'
# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

x_train, _, _, _, _ = read_and_split_data()

tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(x_train)

# Define the FastAPI app
app = FastAPI()

def format_response(response: str) -> dict:
  return {"input": response}

# Define a model input schema
class ModelInput(BaseModel):
    text: object

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def predict(input: ModelInput):
    try:
        prediction = loaded_model.predict(pad_sequences(tokenizer.texts_to_sequences([input.text]), maxlen=10))
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
