# %%writefile streamlit_food.py
import streamlit as st
from fastai.vision.all import *
import PIL

# Load the trained model
learn = load_learner("export.pkl")

# Streamlit UI
st.title("🍽️ Mongolian Food Classifier 🍽️")
st.write("Upload an image of Mongolian food, and the AI will classify it!")

# File uploader
uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = PIL.Image.open(uploaded_file)
    
    # Display image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Predict
    pred, pred_idx, probs = learn.predict(img)
    
    # Show results
    st.write(f"### Prediction: {pred}")
    st.write(f"Confidence: {probs[pred_idx]:.4f}")
