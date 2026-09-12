# Result Verifier — CGPA Contradiction Checker

Upload one or more university reward/result sheets (PDF or image). The app
extracts each subject's marks/grade, recalculates the CGPA from scratch,
and flags it if it doesn't match the CGPA printed on the sheet. You can also
chat with your uploaded sheets to ask questions about your result.

## Tech
Streamlit • PyMuPDF • Tesseract OCR • FAISS + sentence-transformers (RAG) • Groq (LLM)

## Local setup (Windows)
1. `pip install -r requirements.txt`
2. Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki
3. Create a `.env` file:
   ```
   GROQ_API_KEY=your_key_here
   ```
4. Run: `streamlit run app.py`

## Deploy on Streamlit Community Cloud
1. Push this repo to GitHub
2. New app → point to `app.py`
3. Settings → Secrets → add:
   ```
   GROQ_API_KEY = "your_key_here"
   ```
4. Deploy
