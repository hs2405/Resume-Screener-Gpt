import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from utils import extract_text_from_pdf, smart_feedback

# Local label map
label_map = {
    0: "Software Engineer",
    1: "Data Scientist",
    2: "Product Manager",
    3: "UI/UX Designer",
    4: "DevOps Engineer"
}

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
    model = AutoModelForSequenceClassification.from_pretrained("bhadresh-savani/bert-base-uncased-emotion")
    return tokenizer, model

def classify_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, k=3)
    return [(label_map.get(idx.item(), "Unknown"), prob.item()) for idx, prob in zip(top_indices[0], top_probs[0])]

st.set_page_config(page_title="Resume Screener GPT (Offline)", layout="centered")
st.title("🤖 Offline Resume Screener (No API, GPT-like)")
st.write("Upload your PDF resume or paste content to get top 3 predicted roles and suggestions.")

pdf_file = st.file_uploader("📄 Upload PDF Resume", type=["pdf"])
resume_text = ""

if pdf_file:
    resume_text = extract_text_from_pdf(pdf_file)
    st.success("✅ Text extracted from PDF.")
else:
    resume_text = st.text_area("Or paste resume content manually", height=200)

if st.button("🔍 Classify Resume"):
    if not resume_text.strip():
        st.warning("Please upload or enter resume text.")
    else:
        tokenizer, model = load_model()
        predictions = classify_text(resume_text, tokenizer, model)
        st.subheader("🎯 Top 3 Predicted Roles:")
        for role, prob in predictions:
            st.markdown(f"- **{role}** ({prob * 100:.2f}%)")
        st.subheader("💬 GPT-like Feedback:")
        st.info(smart_feedback(predictions[0][0]))
