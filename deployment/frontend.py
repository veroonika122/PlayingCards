import os
import streamlit as st
from PIL import Image
import requests

# Set page config
st.set_page_config(
    page_title="Playing Cards Classifier",
    page_icon="🎴",
    layout="wide"
)

# Title and description
st.title("Playing Cards Classifier")
st.markdown("""
This app classifies playing cards using a deep learning model.
Upload an image of a playing card to get started!
""")

# File uploader
uploaded_file = st.file_uploader("Choose a playing card image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Create two columns
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button('Classify Card'):
            # Convert image to bytes
            img_bytes = uploaded_file.getvalue()
            
            # Send to backend
            files = {'file': ('image.jpg', img_bytes, 'image/jpeg')}
            response = requests.post('http://localhost:8000/classify/', files=files)
            
            if response.status_code == 200:
                result = response.json()
                st.success(f"Prediction: {result['prediction']}")
                
                # Display probabilities
                st.write("Top 5 Predictions:")
                for card, prob in sorted(result['probabilities'].items(), key=lambda x: float(x[1]), reverse=True)[:5]:
                    st.write(f"{card}: {float(prob):.2f}%")
            else:
                st.error("Error in classification")
