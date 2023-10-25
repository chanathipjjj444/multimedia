import streamlit as st
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
import time
fig = plt.figure()



img_size = 224
optimizer = Adam(learning_rate=0.001)
model = tf.keras.models.load_model("best_model.h5")
class_labels = {0: 'Green', 1: 'Overripe', 2: 'Ripe'}

with open("custom.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
st.title(' Banana classifier')

def main():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        img_array = np.array(image)
        
        # Display the resized image
        st.image(img_array, caption="Uploaded Image", use_column_width=True)
        class_btn = st.button("Classify")
        if class_btn:
            if uploaded_file is None:
                st.write("Error!!, please upload an image")
            else:
                with st.spinner('Model working....'):
                    plt.imshow(image)
                    plt.axis("off")
                    predictions = predict(img_array)
                    st.success('Classified')
                    st.write(predictions)

                    
                    
                    

def predict(img_array):
    resized_image = cv2.resize(img_array, (224, 224))
    resized_image = resized_image / 127.5
    img = np.expand_dims(resized_image, axis=0)

    # Make predictions using the loaded model
    predictions = model.predict(img)
    
    # Define emotion labels
    emotion_labels = ["Green","Overripe","Ripe"]

    # Display the predicted emotion
    predicted_class = tf.argmax(predictions, axis=1).numpy()[0]
    predicted_label = class_labels[predicted_class]
    time.sleep(1)
    st.latex(f'Predicted Ripeness: ***** {predicted_label} *****')
    
    for i in range(len(predictions)):
        predictions[i] = predictions[i] * 100
    
    st.title("_________________________________")


if __name__ == "__main__":
    main()