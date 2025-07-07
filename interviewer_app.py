import streamlit as st
import os
import io
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import boto3
from botocore.exceptions import NoCredentialsError
import hashlib
import pandas as pd

# --- Authentication Helpers ---
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Dummy in-memory user DB (for testing)
if 'user_db' not in st.session_state:
    st.session_state.user_db = {
        'testuser': make_hashes('testpass')  # Default user
    }

# --- Login / Signup Interface ---
st.title("🔐 Login or Sign Up")

auth_action = st.selectbox("Choose Action", ["Login", "Sign Up"])
username = st.text_input("Username")
password = st.text_input("Password", type="password")

if auth_action == "Sign Up":
    if st.button("Create Account"):
        if username in st.session_state.user_db:
            st.warning("Username already exists. Try a different one.")
        else:
            st.session_state.user_db[username] = make_hashes(password)
            st.success("Account created! Please log in.")

elif auth_action == "Login":
    if st.button("Login"):
        if username in st.session_state.user_db and check_hashes(password, st.session_state.user_db[username]):
            st.success(f"Welcome back, {username}!")
            st.session_state['authenticated'] = True
            st.session_state['username'] = username
        else:
            st.error("Invalid username or password.")

# Stop if not authenticated
if not st.session_state.get("authenticated", False):
    st.stop()

# --- Main App: AI Interviewer ---
st.set_page_config(page_title="AI Interviewer", layout="wide")
st.title("🤖 AI Interviewer")

# --- OpenAI API Key Setup ---
os.environ["OPENAI_API_KEY"] = "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"

# --- NLTK Download ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    st.success("NLTK data loaded successfully.")
except Exception as e:
    st.error(f"Error loading NLTK data: {e}")
    st.stop()

# --- LangChain and OpenAI ---
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.7)

# --- Boto3 S3 ---
s3_client = None
try:
    s3_client = boto3.client('s3')
    st.sidebar.success("AWS S3 client initialized.")
except NoCredentialsError:
    st.sidebar.warning("AWS credentials not found. S3 disabled.")
    s3_client = None

# --- Helpers ---
def extract_text_from_pdf(pdf_file):
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        return ''.join([page.extract_text() or '' for page in reader.pages])
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def preprocess_text(text):
    if not text:
        return ""
    text = re.sub(r'[^a-z0-9\s]', '', text.lower())
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return " ".join([w for w in tokens if w.isalpha() and w not in stop_words])

def upload_to_s3(bucket_name, file_object, object_name):
    if not s3_client:
        st.error("S3 client not available.")
        return False
    try:
        s3_client.upload_fileobj(file_object, bucket_name, object_name)
        st.success(f"Uploaded to s3://{bucket_name}/{object_name}")
        return True
    except Exception as e:
        st.error(f"S3 upload failed: {e}")
        return False

# --- LangChain Chains ---
question_template = """
You are an expert interviewer. Based on the following job description and candidate resume,
generate 5 interview questions. These should be technical, behavioral, and situational.

Job Description:
{job_description}

Candidate Resume Summary:
{resume_summary}

Generated Questions:
"""
question_prompt = PromptTemplate(
    input_variables=["job_description", "resume_summary"],
    template=question_template
)
question_chain = LLMChain(llm=llm, prompt=question_prompt, output_key="interview_questions")

evaluation_template = """
You are an expert interviewer. Evaluate the candidate's answer.
Provide feedback, strengths/weaknesses, and score out of 10.

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
evaluation_prompt = PromptTemplate(
    input_variables=["job_description", "resume_summary", "question", "answer"],
    template=evaluation_template
)
evaluation_chain = LLMChain(llm=llm, prompt=evaluation_prompt, output_key="evaluation_result")

# --- UI Sidebar ---
st.sidebar.header("Upload Resume & Job Description")

job_description = st.sidebar.text_area("Paste Job Description:", height=300)
uploaded_resume = st.sidebar.file_uploader("Upload Resume (PDF)", type=["pdf"])
s3_bucket_name = st.sidebar.text_input("S3 Bucket Name (optional):")

# --- Logout Option ---
st.sidebar.markdown("---")
if st.sidebar.button("Logout"):
    st.session_state.authenticated = False
    st.experimental_rerun()

# --- Main Logic ---
if not job_description or not uploaded_resume:
    st.info("Please upload resume and job description to continue.")
    st.stop()

with st.spinner("Processing..."):
    resume_text = extract_text_from_pdf(uploaded_resume)
    if not resume_text:
        st.stop()
    resume_summary = preprocess_text(resume_text)
    jd_summary = preprocess_text(job_description)

    if not resume_summary or not jd_summary:
        st.error("Failed to process resume or job description.")
        st.stop()
    st.success("Text processed!")

    if s3_client and s3_bucket_name:
        st.subheader("S3 Upload")
        if st.button("Upload Resume to S3"):
            uploaded_resume.seek(0)
            upload_to_s3(s3_bucket_name, uploaded_resume, f"resumes/{uploaded_resume.name}")
        if st.button("Upload Job Description to S3"):
            jd_bytes = io.BytesIO(job_description.encode('utf-8'))
            upload_to_s3(s3_bucket_name, jd_bytes, f"job_descriptions/job.txt")

st.subheader("Generated Interview Questions")

if st.button("Generate Questions"):
    if 'questions_generated' not in st.session_state:
        st.session_state.questions_generated = False
        st.session_state.generated_questions = []
        st.session_state.evaluation_results = {}
        st.session_state.answers = {}

    with st.spinner("Generating..."):
        try:
            response = question_chain.run(
                job_description=jd_summary,
                resume_summary=resume_summary
            )
            questions_list = [q.strip() for q in response.split('\n') if q.strip() and q.strip()[0].isdigit()]
            st.session_state.generated_questions = questions_list
            st.session_state.questions_generated = True
            st.success("Questions ready!")
        except Exception as e:
            st.error(f"Error generating questions: {e}")

if st.session_state.get('questions_generated', False):
    st.write("Answer the questions below:")
    for i, question in enumerate(st.session_state.generated_questions):
        st.markdown(f"**Q{i+1}:** {question}")
        st.session_state.answers[i] = st.text_area(f"Your Answer (Q{i+1}):", key=f"ans_{i}", height=150)

        if st.session_state.answers[i]:
            if st.button(f"Evaluate Answer {i+1}", key=f"eval_btn_{i}"):
                with st.spinner("Evaluating..."):
                    try:
                        evaluation = evaluation_chain.run(
                            job_description=jd_summary,
                            resume_summary=resume_summary,
                            question=question,
                            answer=st.session_state.answers[i]
                        )
                        st.session_state.evaluation_results[i] = evaluation
                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

        if i in st.session_state.evaluation_results:
            st.markdown(f"**Evaluation for Q{i+1}:**")
            st.write(st.session_state.evaluation_results[i])
            st.markdown("---")

elif st.session_state.get('questions_generated', False):
    st.warning("No questions generated. Try refining your input.")

st.markdown("---")
st.caption("© 2025 AI Interviewer | Built with Streamlit, OpenAI, LangChain, NLTK, Boto3")
