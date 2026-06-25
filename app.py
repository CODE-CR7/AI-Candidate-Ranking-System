import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
st.title("AI Candidate Ranking System")

job_desc = st.text_area("Enter Job Description")

resumes = st.text_area(
    "Enter Candidate Resumes (one per line)"
)

if st.button("Rank Candidates"):
    resume_list = [r.strip() for r in resumes.split("\n") if r.strip()]
    

    docs = [job_desc] + resume_list

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(docs)

    scores = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:]
    )[0]

    ranked = sorted(
        zip(resume_list, scores),
        key=lambda x: x[1],
        reverse=True
    )
    best_candidate, best_score = ranked[0]

    candidate_name = best_candidate.split(":")[0]

    st.success(
    f"🏆 Best Candidate: {candidate_name}\n\nMatch Score: {best_score*100:.1f}%"
   )
    st.subheader("Ranked Candidates")

    for i, (resume, score) in enumerate(ranked, 1):
        match_percent = score * 100
        
        st.write(
           f"🏅 Rank {i} | Match: {match_percent:.1f}% | {resume}"
    )
    st.subheader("Skill Match Analysis")

    jd_words = set(job_desc.lower().split())

    for resume, score in ranked:
     candidate_name = resume.split(":")[0]

    resume_words = set(resume.lower().split())
    matched = jd_words.intersection(resume_words)

    st.write(f"👤 {candidate_name}")
    st.write(f"✅ Matched Skills: {', '.join(matched)}")
    st.write("---")
    uploaded_files = st.file_uploader(
    "Upload Resumes",
    type=["pdf"],
    accept_multiple_files=True
    )