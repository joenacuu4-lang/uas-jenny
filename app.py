
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI by Eva Jennysha")
st.title("🌿 AI Plant Disease Detection")
st.markdown("### UAS Artificial Intelligence")
st.info("**Nama:** Eva Jennysha | **Kelas:** Informatika 4C")
st.divider()

st.write("Ini adalah prototype aplikasi AI buatan saya.")
uploaded_file = st.file_uploader("Upload Foto...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((64, 64))
    st.image(image, caption='Input', use_column_width=True)
    
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    model = tf.keras.models.load_model('model_uas_eva.h5')
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    
    class_names = ['Healthy', 'Rust', 'Mildew']
    
    result_idx = np.argmax(score)
    confidence = 100 * np.max(score)
    
    st.success(f"Hasil: **{class_names[result_idx]}**")
    st.progress(float(np.max(score)))
