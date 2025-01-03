import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load the pre-trained MNIST model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_model.keras")
    return model

model = load_model()

# Function to process the image (resize, convert to grayscale, etc.)
def process_image(image):
    # Convert to grayscale
    image = image.convert("L")
    # Resize the image to 28x28 pixels (MNIST image size)
    image = image.resize((28, 28))
    # Invert the image (MNIST images are white on black, but we want black on white)
    image = ImageOps.invert(image)
    # Normalize to [0, 1]
    image = np.array(image) / 255.0
    # Reshape for the model (batch_size, height, width, channels)
    image = np.expand_dims(image, axis=-1)
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

# Create Streamlit layout
st.title("Handwritten Digit Recognition")
st.write("Draw a digit on the canvas below and the model will predict it!")

# Display the canvas for drawing
canvas_result = st_canvas(
    width=280, 
    height=280, 
    drawing_mode="freedraw", 
    key="canvas"
)

if canvas_result.image_data is not None:
    # Convert the drawn image to a PIL image
    drawn_image = Image.fromarray(canvas_result.image_data.astype("uint8"))
    
    # Process the image and make a prediction
    processed_image = process_image(drawn_image)
    prediction = model.predict(processed_image)
    
    # Get the predicted digit
    predicted_digit = np.argmax(prediction)

    # Display the predicted digit
    st.image(drawn_image, caption="Your Drawing", use_column_width=True)
    st.write(f"Prediction: {predicted_digit}")
