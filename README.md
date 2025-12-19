# CodeGenAI and Explainer
Infosys AI Internship 6.0 – CodeGenie Project

# Overview

CodeGenAI and Explainer is a local AI-powered assistant designed to generate, explain, review, and fix code.
It was developed as part of the Infosys Internship 6.0 (B2) – CodeGenie: AI Explainer and Code Generator program.

The application combines Streamlit, Ollama, and OCR technologies to allow users to interact with code through chat, files, and images, all while keeping processing local and private.

# Internship Details

Organization: Infosys (Springboard)

Internship: Internship 6.0 (B2) – CodeGenie

Duration: September 1, 2025 – November 5, 2025

Domain: Artificial Intelligence

Status: Successfully Completed

# Key Features

💬 Interactive AI Chat
Ask questions, generate code, get explanations, or request fixes in real time.

📎 File Upload and Analysis
Upload images, PDFs, DOCX, TXT, or CSV files for code extraction and analysis.

🖼️ OCR Support
Extracts code and text from images using Tesseract OCR with preprocessing for accuracy.

🤖 Local LLM with Ollama
Runs a local large language model for fast, private, on-device code understanding.

🛠️ Code Review and Fixes
Automatically checks uploaded code for syntax errors, logic issues, and potential improvements.

# How It Works
1. Streamlit Frontend

Provides a simple web interface.

Displays chat history, uploaded files, and AI responses.

Handles dynamic user input and file uploads.

Shows extracted OCR text before sending it to the LLM.

2. OCR with pytesseract

Preprocesses images using grayscale, thresholding, and filtering.

Supports formats: JPG, JPEG, PNG, BMP, TIFF.

Displays extracted text for user verification.

3. File Parsing

PDF: pdfplumber

DOCX: python-docx

CSV / TXT: pandas or native file reading

4. Ollama LLM

Runs a local large language model.

Analyzes, explains, and corrects code.

Supports streaming responses for live feedback.

Works offline once models are installed.

# Installation
Clone the Repository
git clone https://github.com/CheboluGayatri/CodeGenAiand_Explainer.git
cd codegen-ai

Install Python Dependencies
pip install -r requirements.txt

Install Tesseract OCR (Required for Images)

Windows: Download from official Tesseract site

Linux:

sudo apt install tesseract-ocr


macOS:

brew install tesseract

Install Ollama (Optional but Recommended)
brew install ollama
ollama pull gemma:2b
ollama serve


For other platforms, refer to the Ollama documentation.

Usage
streamlit run app.py


Open your browser at http://localhost:8501

Start a new chat or continue an existing one

Ask coding questions or upload files/images

View AI-generated explanations and fixes

# Supported File Types
File Type	Extraction Method

Images	OCR via pytesseract

PDF	pdfplumber

DOCX	python-docx

TXT / CSV	pandas / native read
# Project Structure
app.py            # Main Streamlit application

uploads/          # Uploaded files storage

chats.json        # Chat session history

requirements.txt  # Project dependencies
