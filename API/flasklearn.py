from flask import Flask, request, jsonify
from tensorflow import keras
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load the model
model = keras.models.load_model("model.keras")

# Peprocess the image
def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure an image file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']

        # Load the image
        image = Image.open(file)

        # Preprocess the image
        processed_image = preprocess_image(image, target_size=(128, 128))  # Adjust size as per your model

        # Make prediction
        prediction = model.predict(processed_image)
        class_idx = np.argmax(prediction, axis=1)[0]

        class_name = "Dog" if class_idx == 1 else "Cat"

        return jsonify({'prediction': class_name})
    
    except Exception as e:
        # Handle exceptions and return an error message
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
