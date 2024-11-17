import pandas as pd
import joblib
import os
# Load the trained model (update the model path)
MODEL_PATH = os.path.abspath('controllers/recommendation/age_group_weight/age_group_weight_preference_model.pkl')

try:
    loaded_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load the model: {e}")

def recommend_frame_weight(age_group):
    try:
        # Create a DataFrame for the new input
        new_input = pd.DataFrame({'Age_Group': [age_group]})

        # Get the predicted probabilities
        new_prediction_prob = loaded_model.predict_proba(new_input)

        # Extract probabilities for each class
        prob_no_use = new_prediction_prob[0][0] * 100  # Probability of not using blue light glasses
        prob_use = new_prediction_prob[0][1] * 100     # Probability of using blue light glasses

        # Recommendation message
        recommendation = (f"Based on your age group, "
                          f"there is a {prob_use:.2f}% chance that you prefer lightweight glasses for comfort. "
                          f"{prob_no_use:.2f}% of people in this age group prefer standard-weight glasses.")

        return {
            "age_group": age_group,
            "recommendation": recommendation,
            "probabilities": {
                "weight_preferences": f"{prob_use:.2f}%",
                "no_weight_preferences": f"{prob_no_use:.2f}%"
            }
        }
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}
