# CodeGenAiand_Explainer
CodeGenesis is a local AI-powered assistant for generating, explaining, and correcting code. It seamlessly integrates Streamlit, Ollama, and OCR technologies to let you interact with code through text or uploaded files/images.

Features

💬 Interactive Chat: Ask questions, get code explanations, or request fixes.

📎 File Upload & Analysis: Upload images, PDFs, DOCX, TXT, or CSV files. Extract code with OCR if needed.

🖼️ OCR Support: Images are preprocessed and scanned for text using pytesseract.

🤖 Local LLM: Powered by Ollama for fast, private, on-device AI code analysis.

🛠️ Code Review & Fixes: Automatically checks for syntax, logic, and security issues in uploaded code snippets.

How It Works
1. Streamlit Frontend

Provides a user-friendly web interface.

Displays chat messages, uploaded files, and LLM responses.

Handles user input and file uploads dynamically.

Renders OCR results for images before sending them to the LLM.

2. OCR with pytesseract

Images are preprocessed (grayscale, contrast, threshold, filtering) for better text recognition.

Supports formats: jpg, jpeg, png, bmp, tiff.

Extracted code/text is displayed in the chat for verification.

3. File Parsing

PDF → pdfplumber

DOCX → python-docx

CSV / TXT → pandas / standard read

4. Ollama LLM

Runs a local large language model (ollama) to analyze and correct code.

Prompts include system instructions + extracted code text.

Supports streaming responses, allowing live feedback in the chat.

Handles hidden prompts for private LLM processing.

Installation

Clone the repository

git clone https://github.com/yourusername/codegen-ai.git
cd codegen-ai


Install dependencies

pip install -r requirements.txt


Install Tesseract OCR (for images)

Windows: Download here

Linux: sudo apt install tesseract-ocr

MacOS: brew install tesseract

Install Ollama (optional for local LLM)

brew install ollama  # macOS
# or follow https://ollama.com/docs for other platforms
ollama pull gemma:2b  # Example model
ollama serve

Usage
streamlit run app.py


Open your browser at http://localhost:8501.

Start a new chat or select an existing one.

Type a coding question or upload a code file/image.

View the AI-generated response and explanation.

Repeat as needed!

Supported File Types
Type	Extraction Method
Images	OCR via pytesseract
PDF	pdfplumber
DOCX	python-docx
TXT / CSV	Pandas / native read
Project Structure
app.py                 # Main Streamlit app
uploads/               # Uploaded files stored here
chats.json             # Session storage for chat history
requirements.txt       # Python dependencies

Notes

The system automatically titles new chats based on the uploaded file or user input.

LLM fallback is available if Ollama is not installed or running.

Large files are truncated to prevent memory issues (up to 12,000 characters for LLM).
