import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Config
st.set_page_config(page_title="Language Detection App", page_icon="🌍", layout="centered")


@st.cache_resource
def load_artifacts():
    model = load_model("SavedModel/simple_rnn_model.h5")
    with open("SavedModel/tokenizer.pkl", "rb") as f:
        tokenizer, label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()



def predict_language(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=80)
    probs = model.predict(padded, verbose=0)[0]
    class_index = np.argmax(probs)
    return label_encoder.inverse_transform([class_index])[0], probs[class_index]


st.title("🌍 Language Detection App")
st.write("Enter a sentence below and the model will predict the language.")
st.divider()
user_text = st.text_area("✍️ Enter text here:", height=120, placeholder="Example: यह एक अच्छा दिन है")
if st.button("🔍 Detect Language"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        language, confidence = predict_language(user_text)

        st.success(f"✅ Predicted Language: **{language}**")
        st.write(f"📈 Confidence: **{confidence:.3f}**")
