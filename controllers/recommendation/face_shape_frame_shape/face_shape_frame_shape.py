
import numpy as np
import os
import joblib 
import pandas as pd

# Load the model
MODEL_PATH = os.path.abspath('controllers/recommendation/face_shape_frame_shape/face_shape_frame_shape.pkl')
try:
    loaded_model = joblib.load(MODEL_PATH)
    y_columns = ["Round", "Oval", "Square", 'Rectangular',"Cat-Eye"]  # Replace with your actual frame style columns
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

def recommend_frame_styles_with_probabilities(face_shape):
    try:
        # Create a DataFrame for the new input (a single face shape)
        new_input = pd.DataFrame({'Face_Shape': [face_shape]})

        # Get the predicted probabilities for each frame style from the loaded model
        new_prediction_prob = loaded_model.predict_proba(new_input)

        # Extract the probabilities for each frame style
        style_probabilities = {}
        for i, style in enumerate(y_columns):
            # Access the probability of the positive class for each frame style
            style_probabilities[style] = new_prediction_prob[i][0][1] * 100  # Probability as a percentage

        return style_probabilities
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
