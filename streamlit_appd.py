import streamlit as st
from transformers import pipeline
import matplotlib.pyplot as plt

# ========== Load models ==========
@st.cache_resource
def load_models():
    vi_model = pipeline("sentiment-analysis", model="NlpHUST/vibert4news-base-cased")
    en_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return vi_model, en_model

vi_model, en_model = load_models()

# ========== Giao diện ==========
st.set_page_config(page_title="Sentiment Analysis App", page_icon="🌍", layout="centered")
st.title("🌍 Sentiment Analysis App")
st.write("Ứng dụng phân tích cảm xúc: **Tiếng Việt** 🇻🇳 | **Tiếng Anh** 🇬🇧")

model_choice = st.selectbox("Chọn model:", ["Tiếng Việt (ViBERT)", "Tiếng Anh (DistilBERT)"])
text = st.text_area("Nhập câu để phân tích:", "PHIM MÙA ĐỎ TỆ QUÁ!")

if st.button("🔍 Phân tích"):
    if model_choice == "Tiếng Việt (ViBERT)":
        result = vi_model(text)
    else:
        result = en_model(text)

    label = result[0]['label']
    score = result[0]['score']

    # 🎨 Hiển thị màu theo nhãn
    if "NEG" in label or label == "NEGATIVE":
        st.error(f"😡 Cảm xúc: **{label}** | Độ tin cậy: {score:.2f}")
        color = "red"
    elif "POS" in label or label == "POSITIVE":
        st.success(f"😊 Cảm xúc: **{label}** | Độ tin cậy: {score:.2f}")
        color = "green"
    else:
        st.warning(f"😐 Cảm xúc: **{label}** | Độ tin cậy: {score:.2f}")
        color = "orange"

    # 📊 Biểu đồ trực quan
    labels = [label, "Khác"]
    values = [score, 1 - score]
    fig, ax = plt.subplots()
    ax.pie(values, labels=labels, autopct='%1.1f%%', colors=[color, "lightgrey"])
    st.pyplot(fig)

    # 📥 Xuất file CSV
    import pandas as pd
    df = pd.DataFrame([{"Text": text, "Label": label, "Score": score}])
    st.download_button(
        label="📥 Tải kết quả CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="sentiment_result.csv",
        mime="text/csv"
    )
