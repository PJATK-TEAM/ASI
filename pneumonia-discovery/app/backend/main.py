import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
from azure.storage.blob import BlobServiceClient
from datetime import datetime
import json, os, uuid
from dotenv import load_dotenv
from autogluon.multimodal import MultiModalPredictor
import tempfile
import pandas as pd


app = FastAPI(title="Pneumonia Detector API", version="1.0.0")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

#load_dotenv()

#blob_service = BlobServiceClient.from_connection_string(os.getenv("AZURE_STORAGE_CONNECTION_STRING"))
#container_client = blob_service.get_container_client("logs")



predictor = MultiModalPredictor.load("../autogluon_model")  # folder, nie .ckpt
class_names = ['Pneumonia', 'Normal']


@app.get("/")
def read_root():
    return {"message": "Welcome to the Pneumonia Detector API!"}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpg", "image/jpeg", "image/png"]:
        return {"error": "Invalid file type. Only JPG and PNG are allowed."}

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        return {"error": "File too large. Maximum size is 5MB."}
    await file.close()

    result = predict_image(content)
    print(result)
    #save_log_to_blob(file.filename, result)

    return {"prediction": result}

@app.get("/history")
def get_classification_history():
    logs = []
    try:
        #blobs = container_client.list_blobs()
        #for blob in blobs:
        #    blob_client = container_client.get_blob_client(blob.name)
        #    content = blob_client.download_blob().readall()
        #    data = json.loads(content)

        #    logs.append({
        #        "file_name": data.get("filename"),
        #        "timestamp": data.get("timestamp"),
        #        "result": data.get("prediction")
        #    })

        # Posortuj po najnowszych
        logs = sorted(logs, key=lambda x: x["timestamp"], reverse=True)
        return logs
    except Exception as e:
        return {"error": f"Failed to fetch history: {str(e)}"}


def predict_image(image_bytes: bytes) -> dict:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(image_bytes)
        image_path = tmp.name

    df = pd.DataFrame({"image": [image_path]})

    probs = predictor.predict_proba(df, as_pandas=True, as_multiclass=True)
    row = probs.iloc[0]  # np. {'0': 0.0001, '1': 0.9999}

    # Mapowanie indeksów na klasy
    index_to_class = {0: "Normal", 1: "Pneumonia"}

    # Przetłumacz indeksy na klasy
    result = {}
    for col in row.index:
        class_name = index_to_class[int(col)]
        result[class_name] = float(row[col])

    return result


def save_log_to_blob(filename: str, prediction: dict):
    log_entry = {
        "filename": filename,
        "timestamp": datetime.utcnow().isoformat(),
        "prediction": prediction
    }

    #blob_name = f"{uuid.uuid4()}.json"
    #blob_data = json.dumps(log_entry)

    #container_client.upload_blob(
    #    name=blob_name,
    #    data=blob_data,
    #    overwrite=True,
    #    content_type="application/json"
    #)