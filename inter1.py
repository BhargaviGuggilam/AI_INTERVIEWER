import streamlit as st
import hashlib
import os
import io
import PyPDF2
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI

# -------------------------------
# 🧠 AUTH HELPERS
# -------------------------------
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Dummy in-memory DB
if 'user_db' not in st.session_state:
    st.session_state.user_db = {'admin': make_hashes('admin')}
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

# -------------------------------
# 🧠 LOGIN / SIGNUP
# -------------------------------
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("🔐 Welcome to AI Interviewer Login Portal")

auth_choice = st.selectbox("Choose Option", ["Login", "Sign Up"])
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if auth_choice == "Sign Up":
    if st.button("Create Account"):
        if username in st.session_state.user_db:
            st.warning("Username already exists!")
        else:
            st.session_state.user_db[username] = make_hashes(password)
            st.success("Account created. Please login.")

elif auth_choice == "Login":
    if st.button("Login"):
        if username in st.session_state.user_db and check_hashes(password, st.session_state.user_db[username]):
            st.success(f"Welcome, {username}!")
            st.session_state.authenticated = True
        else:
            st.error("Invalid credentials!")

if not st.session_state.authenticated:
    st.stop()

# -------------------------------
# 🔧 NLTK SETUP
# -------------------------------
try:
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK: {e}")
    st.stop()

# -------------------------------
# 🤖 LLM SETUP (OpenAI)
# -------------------------------
os.environ["OPENAI_API_KEY"] = "sk-..."  # <--- Use your actual key here
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# -------------------------------
# 📂 TEXT FUNCTIONS
# -------------------------------
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return "".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        st.error(f"Failed to extract PDF: {e}")
        return None

def preprocess(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    return " ".join([t for t in tokens if t.isalpha() and t not in stop_words])

# -------------------------------
# 📋 PROMPTS
# -------------------------------
question_prompt = PromptTemplate(
    input_variables=["job_description", "resume_summary"],
    template="""
You're an expert interviewer. Generate 5 interview questions (technical, behavioral, situational)
based on the job description and candidate resume.

Job Description:
{job_description}

Candidate Resume:
{resume_summary}

Interview Questions:
"""
)

behavioral_prompt = PromptTemplate(
    input_variables=[],
    template="""
Generate 5 behavioral interview questions to assess a candidate’s personality traits,
teamwork, conflict resolution, and leadership.
"""
)

professional_prompt = PromptTemplate(
    input_variables=[],
    template="""
Generate 5 situational/professional interview questions that test decision-making,
prioritization, leadership under pressure, and communication skills.
"""
)

evaluation_prompt = PromptTemplate(
    input_variables=["job_description", "resume_summary", "question", "answer"],
    template="""
You're an expert interviewer. Evaluate the candidate's answer to the question below.
Provide feedback with strengths, weaknesses, and a score out of 10.

Job Description:
{job_description}

Resume Summary:
{resume_summary}

Question:
{question}

Answer:
{answer}

Evaluation:
"""
)

# LangChain Chains
resume_chain = LLMChain(llm=llm, prompt=question_prompt)
behavioral_chain = LLMChain(llm=llm, prompt=behavioral_prompt)
professional_chain = LLMChain(llm=llm, prompt=professional_prompt)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt)

# -------------------------------
# 🗂️ TABS FOR INTERVIEW MODES
# -------------------------------
tab1, tab2, tab3 = st.tabs(["📄 Resume-Based", "🗣️ Behavioral", "💼 Professional"])

# -------------------------------
# 📄 Resume-Based Interview
# -------------------------------
with tab1:
    st.header("📄 Resume-Based Interview")

    job_desc = st.text_area("Paste Job Description", height=200)
    resume_file = st.file_uploader("Upload Resume PDF", type=["pdf"])

    if job_desc and resume_file:
        with st.spinner("Processing resume..."):
            resume_text = extract_text_from_pdf(resume_file)
            resume_summary = preprocess(resume_text)
            jd_summary = preprocess(job_desc)

        if st.button("Generate Questions"):
            try:
                output = resume_chain.run(job_description=jd_summary, resume_summary=resume_summary)
                questions = [q.strip() for q in output.split("\n") if q.strip()]
                for idx, q in enumerate(questions, 1):
                    st.markdown(f"**Q{idx}:** {q}")
                    ans = st.text_area(f"Your Answer to Q{idx}:", key=f"answer_resume_{idx}", height=120)
                    if ans and st.button(f"Evaluate Q{idx}", key=f"eval_resume_{idx}"):
                        feedback = evaluation_chain.run(
                            job_description=jd_summary,
                            resume_summary=resume_summary,
                            question=q,
                            answer=ans
                        )
                        st.markdown(f"**Evaluation:** {feedback}")
                        st.markdown("---")
            except Exception as e:
                st.error(f"Error: {e}")

# -------------------------------
# 🗣️ Behavioral Interview
# -------------------------------
with tab2:
    st.header("🗣️ Behavioral Questions")
    if st.button("Generate Behavioral Questions"):
        try:
            output = behavioral_chain.run()
            questions = [q.strip() for q in output.split("\n") if q.strip()]
            for idx, q in enumerate(questions, 1):
                st.markdown(f"**Q{idx}:** {q}")
                ans = st.text_area(f"Your Answer to Q{idx}:", key=f"answer_behav_{idx}", height=120)
                st.markdown("---")
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------
# 💼 Professional Interview
# -------------------------------
with tab3:
    st.header("💼 Professional/Situational Questions")
    if st.button("Generate Professional Questions"):
        try:
            output = professional_chain.run()
            questions = [q.strip() for q in output.split("\n") if q.strip()]
            for idx, q in enumerate(questions, 1):
                st.markdown(f"**Q{idx}:** {q}")
                ans = st.text_area(f"Your Answer to Q{idx}:", key=f"answer_prof_{idx}", height=120)
                st.markdown("---")
        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------
# 🔚 Footer
# -------------------------------
st.markdown("---")
st.caption("© 2025 AI Interviewer | Built with Streamlit, OpenAI, LangChain, and NLTK")
