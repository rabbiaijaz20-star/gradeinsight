import streamlit as st
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ---------------- CONFIG ----------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", None)

# On Windows, point pytesseract to your installed tesseract.exe
# Comment this line out if deploying on Streamlit Cloud (Linux finds it automatically)
if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

GRADE_TABLE = [
    (85, 100, 4.0), (80, 84, 3.7), (75, 79, 3.3), (71, 74, 3.0),
    (68, 70, 2.7), (64, 67, 2.3), (61, 63, 2.0), (58, 60, 1.7),
    (54, 57, 1.3), (50, 53, 1.0), (0, 49, 0.0),
]

st.set_page_config(page_title="Result Verifier", layout="centered")
st.title("📄 Result Verifier — CGPA Contradiction Checker")
st.caption("Upload result sheet(s) as PDF or image. The app rechecks the CGPA math and flags mismatches.")

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Add it to .env (local) or Streamlit Secrets (deployed).")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)


# ---------------- TEXT EXTRACTION ----------------
def extract_text_from_pdf(file_bytes):
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page in doc:
        page_text = page.get_text()
        if page_text.strip():
            text += page_text + "\n"
        else:
            # No real text layer -> this page is likely scanned, run OCR
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text += pytesseract.image_to_string(img) + "\n"
    return text


def extract_text_from_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(img)


def extract_text(uploaded_file):
    file_bytes = uploaded_file.read()
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(file_bytes)
    else:
        return extract_text_from_image(file_bytes)


# ---------------- STRUCTURE THE DATA (via Groq) ----------------
def extract_structured_data(raw_text):
    prompt = f"""
You are given raw text extracted from a university result/reward sheet.
Extract ONLY the following as strict JSON, nothing else, no markdown fences:

{{
  "student_name": "string or null",
  "stated_cgpa": number or null,
  "subjects": [
    {{"name": "string", "credit_hours": number, "marks": number or null, "grade_letter": "string or null"}}
  ]
}}

If marks are given as percentage/marks out of 100, put them in "marks".
If only a letter grade is shown, put it in "grade_letter" and leave "marks" null.
If credit hours are missing for a subject, use 3 as default.

Raw text:
\"\"\"{raw_text}\"\"\"
"""
    response = client.chat.completions.create(
        model="model="openai/gpt-oss-120b",",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


# ---------------- CGPA CALCULATOR ----------------
def marks_to_gpa(marks):
    for low, high, gpa in GRADE_TABLE:
        if low <= marks <= high:
            return gpa
    return 0.0


LETTER_TO_GPA = {
    "A+": 4.0, "A": 4.0, "A-": 3.7, "B+": 3.3, "B": 3.0, "B-": 2.7,
    "C+": 2.3, "C": 2.0, "C-": 1.7, "D+": 1.3, "D": 1.0, "F": 0.0,
}


def compute_cgpa(subjects):
    total_points = 0.0
    total_credits = 0.0
    for s in subjects:
        credit = s.get("credit_hours") or 3
        if s.get("marks") is not None:
            gpa = marks_to_gpa(s["marks"])
        elif s.get("grade_letter"):
            gpa = LETTER_TO_GPA.get(s["grade_letter"].upper().strip(), 0.0)
        else:
            continue
        total_points += gpa * credit
        total_credits += credit
    if total_credits == 0:
        return None
    return round(total_points / total_credits, 2)


def check_contradiction(stated_cgpa, computed_cgpa, tolerance=0.05):
    if stated_cgpa is None or computed_cgpa is None:
        return None
    return abs(stated_cgpa - computed_cgpa) > tolerance


# ---------------- RAG (FAISS + embeddings) ----------------
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")


def build_index(chunks, embedder):
    embeddings = embedder.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype("float32"))
    return index


def retrieve(query, chunks, index, embedder, k=3):
    q_emb = embedder.encode([query]).astype("float32")
    distances, indices = index.search(q_emb, k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ---------------- GROQ EXPLANATION ----------------
def generate_explanation(subjects, stated_cgpa, computed_cgpa, context):
    prompt = f"""
A student's result sheet states CGPA = {stated_cgpa}, but based on the subject
marks/grades, the recalculated CGPA is {computed_cgpa}.

Subject data: {json.dumps(subjects)}

Relevant sheet context:
{context}

Explain in simple, friendly language why there might be a contradiction,
and point to which subject(s) look most likely to be entered incorrectly.
Keep it under 150 words.
"""
    response = client.chat.completions.create(
        model="model="openai/gpt-oss-120b",",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content


# ---------------- STREAMLIT UI ----------------
embedder = load_embedder()

uploaded_files = st.file_uploader(
    "Upload result sheet(s)",
    type=["pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if "all_chunks" not in st.session_state:
    st.session_state.all_chunks = []
    st.session_state.index = None

if uploaded_files:
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"📑 {uploaded_file.name}")

        with st.spinner("Extracting text..."):
            raw_text = extract_text(uploaded_file)

        if not raw_text.strip():
            st.error("Couldn't extract any text from this file. Try a clearer scan/photo.")
            continue

        with st.spinner("Understanding the sheet..."):
            data = extract_structured_data(raw_text)

        if not data or not data.get("subjects"):
            st.error("Couldn't parse subject data from this sheet. The format might be unusual.")
            with st.expander("Show extracted raw text"):
                st.text(raw_text)
            continue

        stated_cgpa = data.get("stated_cgpa")
        computed_cgpa = compute_cgpa(data["subjects"])

        st.write(f"**Student:** {data.get('student_name') or 'Not found'}")
        st.table(data["subjects"])

        col1, col2 = st.columns(2)
        col1.metric("CGPA on sheet", stated_cgpa if stated_cgpa is not None else "Not found")
        col2.metric("Recalculated CGPA", computed_cgpa if computed_cgpa is not None else "N/A")

        mismatch = check_contradiction(stated_cgpa, computed_cgpa)
        if mismatch is True:
            st.error("⚠️ Contradiction found — the stated CGPA does not match the subject marks.")
            with st.spinner("Generating explanation..."):
                explanation = generate_explanation(data["subjects"], stated_cgpa, computed_cgpa, raw_text[:2000])
            st.info(explanation)
        elif mismatch is False:
            st.success("✅ No contradiction — CGPA matches the subject marks.")
        else:
            st.warning("Couldn't compare — missing stated CGPA or subject data.")

        # Add to RAG store for chat
        st.session_state.all_chunks.append(f"File: {uploaded_file.name}\n{raw_text}")

    if st.session_state.all_chunks:
        st.session_state.index = build_index(st.session_state.all_chunks, embedder)

# ---------------- CHAT WITH YOUR SHEETS ----------------
if st.session_state.index is not None:
    st.markdown("---")
    st.subheader("💬 Ask about your uploaded sheet(s)")
    question = st.text_input("Ask a question, e.g. 'Why is my CGPA wrong?'")
    if question:
        with st.spinner("Thinking..."):
            relevant_chunks = retrieve(question, st.session_state.all_chunks, st.session_state.index, embedder)
            context = "\n\n".join(relevant_chunks)
            answer = client.chat.completions.create(
                model="model="openai/gpt-oss-120b",",
                messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer clearly and briefly."}],
                temperature=0.3,
            ).choices[0].message.content
        st.write(answer)
