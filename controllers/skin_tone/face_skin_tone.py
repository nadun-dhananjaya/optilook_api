async def predict_skin_tone():
    return "ss"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import pandas as pd

app = FastAPI()

# # Load the model
MODEL_PATH = os.path.abspath('controllers/skin_tone/skin_tone_model.h5')
# model = tf.keras.models.load_model(MODEL_PATH)

# # Preprocess the image
# def preprocess_image(image_file):
#     image = Image.open(image_file).convert("RGB")
#     image = image.resize((224, 224))  # Resize to model input size
#     image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
#     image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
#     return image_array

# @app.post("/predict/skin_tone")
# async def predict_skin_tone(file: UploadFile = File(...)):
#     try:
#         # Load and preprocess the image
#         image = preprocess_image(file.file)
        
#         # Perform prediction
#         predictions = model.predict(image)
#         predicted_class = np.argmax(predictions, axis=1)[0]
#         confidence = predictions[0][predicted_class]

#         # Class labels (update as per your model's output)
#         class_labels = ["Fair", "Medium", "Dark"]  # Example class labels

#         return JSONResponse({
#             "predicted_class": class_labels[predicted_class],
#             "confidence": f"{confidence * 100:.2f}%",
#             "class_probabilities": {class_labels[i]: f"{prob * 100:.2f}%" for i, prob in enumerate(predictions[0])}
#         })
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)

