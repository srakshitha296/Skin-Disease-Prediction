import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io


model = load_model('skin_disease_model.h5')


class_names = ['FU-nail-fungus', 'FU-ringworm', 'VI-shingles', 'BA-impetigo', 'FU-athlete-foot', 'VI-chickenpox', 'PA-cutaneous-larva-migrans', 'BA-cellulitis']
class_details = {
    'FU-nail-fungus': {'symptoms': 'Itchy nails, discolored nails, thickened nails', 'diagnosis': 'Clinical examination and nail culture'},
    'FU-ringworm': {'symptoms': 'Itchy, red, circular rash with clearer center', 'diagnosis': 'Clinical examination and fungal culture'},
    'VI-shingles': {'symptoms': 'Painful rash with blisters, typically on one side of the body', 'diagnosis': 'Clinical examination and history of chickenpox'},
    'BA-impetigo': {'symptoms': 'Red sores that quickly turn into honey-colored crusts', 'diagnosis': 'Clinical examination and bacterial culture'},
    'FU-athlete-foot': {'symptoms': 'Itchy, red, scaly rash between the toes', 'diagnosis': 'Clinical examination and fungal culture'},
    'VI-chickenpox': {'symptoms': 'Itchy rash with red spots and blisters', 'diagnosis': 'Clinical examination and history of exposure'},
    'PA-cutaneous-larva-migrans': {'symptoms': 'Red, itchy rash that moves across the skin', 'diagnosis': 'Clinical examination and history of exposure to contaminated soil'},
    'BA-cellulitis': {'symptoms': 'Red, swollen, and painful skin, often with fever', 'diagnosis': 'Clinical examination and bacterial culture'}
}

def predict_skin_disease(img):
    # Preprocess the image
    img = img.resize((150, 150))
    img_array = np.array(img)
    
    # Ensure the array is of type float
    img_array = img_array.astype('float32')
    
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Predict the class
    predictions = model.predict(img_array)
    confidence = np.max(predictions) * 100
    predicted_class = class_names[np.argmax(predictions)]

    if confidence < 75:
        return f"<p style='color:red; font-weight:bold;'>Confidence is too low to make a prediction. Confidence: {confidence:.2f}%</p>"
    else:
        details = class_details[predicted_class]
        return (
            f"<p style='color:blue; font-weight:bold;'>Predicted: {predicted_class} with {confidence:.2f}% confidence</p>"
            f"<p style='font-weight:bold;'>Symptoms: {details['symptoms']}</p>"
            f"<p style='font-weight:bold;'>Diagnosis: {details['diagnosis']}</p>"
        )


st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Project","Prediction"])

if(app_mode=="Home"):
    st.header("SKIN DISEASE RECOGNITION")
    image_path = "image.png"
    st.image(image_path)

elif(app_mode=="About Project"):
    st.header("SKIN DISEASE RECOGNITION")   
    st.subheader("About skin disease")
    st.text("write something................................................................................................................................")
elif(app_mode=="Prediction"):
    st.title("Skin Disease Classifier")
    st.write("Upload an image of a skin condition and get a prediction with confidence percentage.")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")

        result = predict_skin_disease(image)
        st.markdown(result, unsafe_allow_html=True)

