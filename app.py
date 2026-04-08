
import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

st.set_page_config(page_title="AI Resume Screener", layout="wide")
st.title("📄 AI Resume Screener (Pro)")

col1, col2 = st.columns(2)

with col1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

with col2:
    jd = st.text_area("Paste Job Description")

if st.button("Analyze"):
    if resume_file and jd:
        resume_text = extract_text(resume_file)
        tfidf = TfidfVectorizer(stop_words='english')
        vectors = tfidf.fit_transform([resume_text, jd])
        score = cosine_similarity(vectors[0], vectors[1])[0][0]
        st.success(f"Match Score: {round(score*100,2)}%")
    else:
        st.warning("Please upload resume and enter job description")
