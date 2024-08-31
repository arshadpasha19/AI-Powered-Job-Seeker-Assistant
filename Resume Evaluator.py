import streamlit as st
import PyPDF2 as pdf
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "Enter your OpenAI API key"
os.environ['LANGCHAIN_API_KEY'] = "Enter your Langchain API Key"

def input_pdf_text(uploaded_file):
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# Prompt Template
input_prompt_template = """
**Hey Act Like a skilled or very experienced ATS (Application Tracking System)**
with a deep understanding of the tech field, software engineering, data science, data analysis,
and big data engineering. Your task is to evaluate the resume based on the given job description.
Consider a competitive job market and provide the best assistance for improving the resumes.

**My Resume:** {text}
**My Experience:** {experience}
**Role:** {role}
**Job Description:** {jd}

**Desired Output:**

* **ATS Score:** Percentage matching based on JD and missing keywords (high accuracy).
* **Missing Keywords:** List of relevant keywords from the JD not found in the resume.
* **Can Select (Apply/Hold):** Recommendation based on the ATS score (apply or consider improving the resume).
* **Improvement Tips:** Bullet points with actionable advice to enhance the resume's fit for the job.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant, please answer to user queries"),
    ("user", input_prompt_template)
])

llm = ChatOpenAI(model="gpt-3.5-turbo")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

# Streamlit app
st.title(':rainbow[RESUME EVALUATOR]')
#st.text("Improve Your Resume where ATS Efficiency Meets Accuracy: Unveiling LLM-Based ATS Resume Scoring Technology")

name = st.text_input("Enter Your Name:")
roleoptions = ["--Select--", "Data Analyst", "Data Scientist", "Machine Learning Engineer", "Data Engineer", "Software Engineer", "Full Stack Engineer"]
role = st.selectbox("Select a role: ", roleoptions)
options = ["--Select--", "Fresher", "1-2 Years", "2-3 Years", "3-4 Years", "4-5 Years", "5-6 Years", "7-10 Years"]
experience = st.selectbox("Select an option:", options)
jd = st.text_area("Paste the Job Description")
uploaded_file = st.file_uploader("Upload Your Resume", type="pdf", help="Please upload the PDF")

submit = st.button("Submit")

if submit:
    if uploaded_file is not None:
        resume_text = input_pdf_text(uploaded_file)
        input_data = {"text": resume_text, "jd": jd,"role": role, "experience": experience}
        response = chain.invoke(input_data)
        st.subheader("ATS Evaluation Result")
        st.write(response)
    else:
        st.error("Please upload your resume")
