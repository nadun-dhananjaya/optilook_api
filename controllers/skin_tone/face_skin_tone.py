
from PIL import Image
import numpy as np
import tensorflow as tf
import os

# Load the model
MODEL_PATH = os.path.abspath('controllers/skin_tone/skin_tone_model_new.h5')
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(file):
    """
    Preprocess the uploaded image file for prediction.
    """
    # Read the file content from SpooledTemporaryFile
    image = Image.open(file).convert("RGB")
    image = image.resize((224, 224))  # Resize to model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

async def predict_skin_tone(file):
    """
    Predicts the skin tone based on the uploaded image.

    Parameters:
        file: File, uploaded image.

    Returns:
        dict: Prediction results with class and probabilities.
    """
    try:
        # Preprocess the image
        image = preprocess_image(file)
        
        # Perform prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class]

        # Class labels (update as per your model's output)
        class_labels = ["Cool", "Neutral", "Warm"]

        return {
            "predicted_class": class_labels[predicted_class],
            "confidence": f"{confidence * 100:.2f}%",
            "class_probabilities": {class_labels[i]: f"{prob * 100:.2f}%" for i, prob in enumerate(predictions[0])}
        }
    except Exception as e:
        return {"error": str(e)}
