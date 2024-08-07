from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



# Define the FastAPI app
app = FastAPI()

def format_response(response: str) -> dict:
  return {"input": response}

# Define a model input schema
class ModelInput(BaseModel):
    text: object


# Load the model 
def load_model(path: str):
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")
    return model

# Path to your model file
model_path = r"notebooks\mlruns\1\3ff080f0cf984f4daa3b210ad8d2cdad\artifacts\KNN\model.pkl"
# Load the model
model = load_model(model_path)

@app.post("/predict")
def predict(input: ModelInput):
    try:
        tfidf = TfidfVectorizer()
        # tokenizer = Tokenizer(num_words=2000)
        # tokenizer.fit_on_texts(input.text)
        # input.text = tokenizer.texts_to_sequences(input.text)
        # input.text = pad_sequences(input.text, maxlen=100)
        
        # Transform the input text using the loaded vectorizer
        X = tfidf.fit_transform([input.text]).toarray()
        
        # data = np.array(input.text).reshape(1, -1)  # Reshape data for prediction
        prediction = model.predict(X)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
