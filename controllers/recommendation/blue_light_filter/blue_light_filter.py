import pandas as pd
import joblib
import os
# Load the trained model (update the model path)
MODEL_PATH = os.path.abspath('controllers/recommendation/blue_light_filter/logistic_blue_light_model.pkl')

try:
    loaded_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

def recommend_blue_light(screen_time):
   
    try:
        # Create a DataFrame for the new input
        new_input = pd.DataFrame({'Screen_Time': [screen_time]})

        # Get the predicted probabilities
        new_prediction_prob = loaded_model.predict_proba(new_input)

        # Extract probabilities for each class
        prob_no_use = new_prediction_prob[0][0] * 100  # Probability of not using blue light glasses
        prob_use = new_prediction_prob[0][1] * 100     # Probability of using blue light glasses

        # Recommendation message
        recommendation = (f"Based on your screen time, "
                          f"there is a {prob_use:.2f}% chance that you may need blue light blocking glasses. "
                          f"{prob_no_use:.2f}% of people with similar screen time do not use blue light blocking glasses.")

        return {
            "screen_time": screen_time,
            "recommendation": recommendation,
            "probabilities": {
                "use_blue_light_blocking": f"{prob_use:.2f}%",
                "no_need_for_blue_light_blocking": f"{prob_no_use:.2f}%"
            }
        }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
