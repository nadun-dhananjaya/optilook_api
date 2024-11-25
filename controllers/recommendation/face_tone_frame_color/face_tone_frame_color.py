
import numpy as np
import os
import joblib 
import pandas as pd

# Load the trained model 
MODEL_PATH = os.path.abspath('controllers/recommendation/face_tone_frame_color/face_color_frame_color.pkl')
try:

    loaded_model = joblib.load(MODEL_PATH)
    y_columns = ["Black", "Gold", "Orange", "Blue", "Pink","Green"]  
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

def recommend_frame_color_with_probabilities(face_color):
    try:
        # Create a DataFrame for the new input (a single face shape)
        new_input = pd.DataFrame({'Skin_Tone': [face_color]})

        # Get the predicted probabilities for each frame color from the loaded model
        new_prediction_prob = loaded_model.predict_proba(new_input)

        # Extract the probabilities for each frame color
        color_probabilities = {}
        for i, color in enumerate(y_columns):  # Assuming y_columns has the names of frame colors
            color_probabilities[color] = new_prediction_prob[i][0][1] * 100  # Probability as a percentage

        return color_probabilities
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
