import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.preprocessing import image
from fastapi.responses import JSONResponse
import tensorflow as tf

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('trained_model.h5')  # Replace with your trained model file path

# Manually define the class names (replace with your actual class names)
class_names = ['aa', 'chi', 'ee']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file into memory (as bytes)
        img_bytes = await file.read()

        # Open the image with PIL or Keras Image from byte stream
        img = image.load_img(io.BytesIO(img_bytes), target_size=(64, 64), color_mode='grayscale')
        
        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        
        # Normalize the image
        img_array = img_array / 255.0
        
        # Add an extra dimension to represent the batch size
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        predictions = model.predict(img_array)
        
        # Get the class index with the highest probability
        predicted_class_index = np.argmax(predictions)
        
        # Get the class name (vocabulary word)
        predicted_class_name = class_names[predicted_class_index]

        # Return the prediction as a JSON response
        return JSONResponse(content={'predicted_class': predicted_class_name})
    
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=400)

# To run the app, use the following command:
# uvicorn your_filename:app --reload
