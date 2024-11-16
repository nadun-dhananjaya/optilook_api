from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from controllers.face_shape.face_shape import predict_face_shape
from controllers.skin_tone.face_skin_tone import predict_skin_tone
from pydantic import BaseModel
from controllers.recommendation.face_shape_frame_shape.face_shape_frame_shape import recommend_frame_styles_with_probabilities  
from controllers.recommendation.face_tone_frame_color.face_tone_frame_color import recommend_frame_color_with_probabilities 
from controllers.recommendation.blue_light_filter.blue_light_filter import recommend_blue_light 
from controllers.recommendation.uv_light_filter.uv_light_filter import recommend_uv_protection
# Allowed origins
origins = ["http://localhost:3000"]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Face Shape Prediction API"}

@app.post("/predict/face_shape")
async def predict_face_shape_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for predicting face shape.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Call the prediction function
        result = await predict_face_shape(file.file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Add additional routes for other models as needed
@app.post("/predict/skin_tone")
async def predict_skin_tone_endpoint(file: UploadFile = File(...)):
    """
    Endpoint for predicting face shape.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    try:
        # Call the prediction function
        result = await predict_skin_tone(file.file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


# Define input structure
class FaceShapeInput(BaseModel):
    face_shape: int

# Add additional routes for other models as needed
@app.post("/predict/face_shape_frame_shape")
async def predict_skin_tone_endpoint(input_data: FaceShapeInput):
    """
    Endpoint for predicting face shape.
    """
    if not input_data:
        raise HTTPException(status_code=400, detail="No frame shape provided")

    try:
        # Call the prediction function
        result = recommend_frame_styles_with_probabilities(input_data.face_shape)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

    # Define input structure
class FaceColorInput(BaseModel):
    face_color: int

# Add additional routes for other models as needed
@app.post("/predict/face_color_frame_color")
async def predict_skin_tone_endpoint(input_data: FaceColorInput):
    """
    Endpoint for predicting face shape.
    """
    if not input_data:
        raise HTTPException(status_code=400, detail="No frame shape provided")

    try:
        # Call the prediction function
        result = recommend_frame_color_with_probabilities(input_data.face_color)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

# Define input schema
class BlueLightInput(BaseModel):
    screen_time: float

@app.post("/recommend/blue_light_glasses")
async def recommend_blue_light_glasses(input_data: BlueLightInput):
    """
    Endpoint for predicting face shape.
    """
    if not input_data:
        raise HTTPException(status_code=400, detail="No frame shape provided")

    try:
        # Call the prediction function
        result = recommend_blue_light(input_data.screen_time)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
class UvInput(BaseModel):
    job_category: float

@app.post("/recommend/uv_filter_glasses")
async def recommend_blue_light_glasses(input_data: UvInput):
    """
    Endpoint for predicting face shape.
    """
    if not input_data:
        raise HTTPException(status_code=400, detail="No frame shape provided")

    try:
        # Call the prediction function
        result = recommend_uv_protection(input_data.job_category)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))