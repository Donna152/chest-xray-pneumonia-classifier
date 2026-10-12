import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.densenet import preprocess_input

st.title('🫁 Chest X-Ray Pneumonia Classifier')
img_size = 256  # DenseNet121 training size
PRETRAINED_MODEL_PATH = "model_transfer.keras"  
CATEGORIES = ["Normal", "Pneumonia"]  # Alphabetical order: Normal=0, Pneumonia=1

# Load the model with caching so it doesn't reload on every button click
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(PRETRAINED_MODEL_PATH)

try:
    model = load_model()
    print('Model Loaded Successfully')
except Exception as e:
    st.error(f"Error loading model file: {e}")

def load_classifier():
    st.subheader("Upload a chest X-ray image to check for pneumonia")
    file = st.file_uploader(label=" ", type=['jpeg', 'jpg', 'png'])

    if file is not None:
        # Load and resize image as RGB (3 channels) to match DenseNet121 requirements
        img = tf.keras.preprocessing.image.load_img(file, target_size=(img_size, img_size), color_mode='rgb')
        new_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Reshape to match batch format: (1, 256, 256, 3)
        new_array = np.expand_dims(new_array, axis=0)
        
        # Apply DenseNet's specific preprocessing
        new_array = preprocess_input(new_array)
        
        # Display uploaded image
        st.image(file, caption="Uploaded Chest X-Ray", use_container_width=True)
        st.write("")
        st.write("")
            
        if st.button("PREDICT"):
            with st.spinner("Analyzing X-ray..."):
                # Making prediction
                prediction = model.predict(new_array)
                print(prediction)
                
                score = prediction[0][0]
                print(score)
                
                # Determine category and confidence
                class_idx = 0 if score < 0.5 else 1
                confidence = (1 - score) * 100 if class_idx == 0 else score * 100
                
                # Formatted prediction string
                preds = (
                    f"{CATEGORIES[class_idx]} — "
                    f"Confidence: {confidence:.2f}% (Raw Score: {score:.4f})"
                )
                
                # Display result cleanly with medical context
                if class_idx == 0:
                    st.success(f"### Prediction: 🟢 {preds}")
                    st.info("The scan exhibits characteristics consistent with normal lung findings.")
                else:
                    st.warning(f"### Prediction: ⚠️ {preds}")
                    st.error("The scan exhibits patterns suggestive of pneumonia. Please consult a qualified medical professional for clinical verification.")

def main():
    load_classifier()

if __name__ == "__main__":
    main()