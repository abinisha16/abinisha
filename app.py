import gradio as gr
import tensorflow as tf
import numpy as np

# Load the pre-trained model
model = tf.keras.models.load_model('mnist_model.h5')  # Load the saved model

# Define the prediction function
def predict_image(image):
    # Preprocess the image to match the model input format
    image = np.array(image) / 255.0  # Normalize the image (same as during training)
    image = image.reshape(1, 28, 28, 1)  # Reshape for CNN input (1, 28, 28, 1)
    
    # Get the model's prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get the class with the highest probability
    
    return predicted_class

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_image,  # Function that will be called for prediction
    inputs=gr.inputs.Image(shape=(28, 28), image_mode='L', invert_colors=True),  # Image input (28x28 grayscale)
    outputs=gr.outputs.Label(num_top_classes=1),  # Display the predicted class
    live=True  # Optional: Set to True to update prediction as you draw
)

# Launch the app
iface.launch()
