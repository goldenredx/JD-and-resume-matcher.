import streamlit as st
import pdfplumber
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text 

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def get_embedding(text):
    return model.encode([text])[0]

def get_similarity(resume_text, jd_text):
    emb1 = get_embedding(resume_text)
    emb2 = get_embedding(jd_text)
    score = cosine_similarity([emb1], [emb2])[0][0]
    return round(score * 100, 2)

def extract_keywords(text, top_n=30):
    words = clean_text(text).split()
    stopwords = set([
        'the','and','for','with','are','you','your','this','that','from',
        'will','have','has','can','our','job','role','work','team','using',
        'able','skills','experience','years','strong','good','knowledge'
    ])
    words = [w for w in words if w not in stopwords and len(w) > 2]
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, c in sorted_words[:top_n]]

def missing_keywords(resume_text, jd_text):
    jd_keywords = set(extract_keywords(jd_text, top_n=40))
    resume_words = set(clean_text(resume_text).split())
    missing = jd_keywords - resume_words
    return list(missing)

## streamlit.
st.set_page_config(page_title="AI Resume-JD Matcher", layout="centered")
st.title("📄 AI Resume ↔ Job Description Matcher")
st.write("Upload your resume and paste the job description to check match score and missing keywords.")

resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
jd_text_input = st.text_area("Paste Job Description here", height=250)

if st.button("Analyze Match"):
    if resume_file is None or not jd_text_input.strip():
        st.warning("Please upload resume and paste job description.")
    else:
        with st.spinner("Analyzing..."):
            resume_raw = extract_text_from_pdf(resume_file)
            resume_clean = clean_text(resume_raw)
            jd_clean = clean_text(jd_text_input)

            score = get_similarity(resume_clean, jd_clean)
            missing = missing_keywords(resume_raw, jd_text_input)

        st.subheader(f"✅ Match Score: {score}%")

        if score >= 75:
            st.success("Strong match!")
        elif score >= 50:
            st.info("Moderate match. Consider adding missing keywords.")
        else:
            st.error("Low match. Resume needs significant tailoring.")

        st.subheader("🔑 Missing Keywords (Top JD terms not in resume)")
        if missing:
            st.write(", ".join(missing[:20]))
        else:
            st.write("No major missing keywords found.")

        with st.expander("View Extracted Resume Text"):
            st.text(resume_raw)