import streamlit as st
import pdfplumber
import joblib
import os
import pandas as pd

# Must be first Streamlit command after imports
st.set_page_config(page_title="Resume Category Predictor", layout="centered")

# ========== Load the Saved Model ==========
model_dir = "files" 

@st.cache_resource
def load_model_components():
    model = joblib.load(os.path.join(model_dir, "logistic_model.pkl"))
    tfidf = joblib.load(os.path.join(model_dir, "tfidf_vectorizer.pkl"))
    label_encoder = joblib.load(os.path.join(model_dir, "label_encoder.pkl"))
    return model, tfidf, label_encoder

model, tfidf, label_encoder = load_model_components()

# ========== Streamlit UI ==========
st.title("📄 Resume Category Predictor")
st.write("Upload one or more PDF resumes to predict the job category automatically.")

uploaded_files = st.file_uploader("Upload Resume PDFs", type=["pdf"], accept_multiple_files=True)

# ========== Process PDFs and Predict ==========
def extract_text_from_pdf(file):
    try:
        with pdfplumber.open(file) as pdf:
            text = ''
            for page in pdf.pages:
                content = page.extract_text()
                if content:
                    text += content + '\n'
        return text.lower()
    except Exception as e:
        st.warning(f"⚠️ Could not read {file.name}: {e}")
        return ''

if uploaded_files:
    results = []

    for file in uploaded_files:
        text = extract_text_from_pdf(file)
        if text.strip():
            X = tfidf.transform([text])
            pred = model.predict(X)
            category = label_encoder.inverse_transform(pred)[0]
            results.append({'Filename': file.name, 'Predicted Category': category})
    
    if results:
        df_results = pd.DataFrame(results)
        st.success("✅ Prediction complete!")
        st.dataframe(df_results, use_container_width=True)
    else:
        st.warning("No valid resumes processed.")
