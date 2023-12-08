import streamlit as st
from PIL import Image
import pickle as pkl

class_list = {'0': 'Negative', '1': 'Neutral', '2': 'Positive'}
with open("styles.css") as f:
    custom_css = f.read()
st.markdown(f"<style>{custom_css}</style>", unsafe_allow_html=True)
st.title('Sentiment analysis from Vietnamese students’ feedback')

image = Image.open('vsfc.jpg')
st.image(image)

input_ec = open('ec_vsfc.pkl', 'rb')
encoder = pkl.load(input_ec)

input_md = open('lrc_vsfc.pkl', 'rb')
model = pkl.load(input_md)

st.header('Write a feedback')
txt = st.text_area('', '')

if txt != '':
    if st.button('Predict'):
        feature_vector = encoder.transform([txt])
        label = str((model.predict(feature_vector))[0])

        st.header('Result')
        st.text(class_list[label])
