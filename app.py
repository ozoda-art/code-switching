import streamlit as st
from model_loader import load_model

# Modelni yuklash
classifier, tokenizer = load_model()

st.title("🤖 Sentiment Analysis for Code-Switched Text")
st.markdown("Rus-Ingliz aralash matnlaringizni ijobiy/salbiy/neytral deb baholaydi")

user_input = st.text_area("✍️ Matnni kiriting:", "I love тебя, but this is ужасно.")

if st.button("Tahlil qilish"):
    import torch

    inputs = tokenizer(user_input, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = classifier(**inputs)
        logits = outputs.logits
        predicted_class = logits.argmax(dim=1).item()

    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    st.subheader("🧠 Model xulosasi:")
    st.success(f"👉 {label_map[predicted_class]}")
