# 🚀 CodeGenAI & Explainer

An AI-powered coding assistant that helps developers **generate, explain, debug, optimize, and review code** using Large Language Models (LLMs). The application also supports **document analysis** and **OCR-based code extraction**, allowing users to analyze code from text files, documents, and images through an intuitive Streamlit interface.

> Developed as part of the **Infosys Springboard Internship 6.0 – CodeGenie: AI Explainer & Code Generator**

---

## 🌐 Live Demo

**Application:** https://codegenaiand-explainer.onrender.com

**GitHub Repository:** https://github.com/CheboluGayatri/CodeGenAiand_Explainer

---

# 📖 Overview

CodeGenAI & Explainer is designed to simplify software development by combining AI-powered code assistance with document and image processing.

Users can:

- Generate code from natural language prompts
- Understand existing code with detailed explanations
- Detect and fix programming errors
- Improve code quality and performance
- Extract code from images using OCR
- Analyze documents containing source code
- Maintain interactive chat history

The application runs locally with **Ollama**, enabling faster inference, improved privacy, and offline AI capabilities.

---

# ✨ Features

## 🤖 AI Code Assistant

- Generate code from natural language descriptions
- Explain code step by step
- Debug syntax and logical errors
- Optimize existing code
- Review code quality and suggest improvements

---

## 📄 Document Analysis

Upload programming-related documents for AI analysis.

Supported formats:

- PDF
- DOCX
- TXT
- CSV

The application extracts the document content and provides explanations, improvements, and code insights.

---

## 🖼 OCR-Based Code Extraction

Extract source code directly from images using **Tesseract OCR**.

Supported image formats:

- PNG
- JPG
- JPEG
- BMP
- TIFF

Image preprocessing is applied to improve OCR accuracy before analysis.

---

## 💬 Chat History

- Interactive AI conversations
- Persistent chat sessions
- Easy access to previous responses

---

## 🔒 Local AI with Ollama

Run Large Language Models locally for:

- Offline inference
- Faster response times
- Enhanced privacy
- No dependency on external AI APIs

---

# 🛠 Technology Stack

| Category | Technology |
|----------|------------|
| Frontend | Streamlit |
| Programming Language | Python |
| AI Model | Ollama |
| OCR Engine | Tesseract OCR |
| Image Processing | Pillow |
| PDF Processing | pdfplumber |
| DOCX Processing | python-docx |
| Data Handling | Pandas |

---

# 📁 Project Structure

```text
CodeGenAiand_Explainer/
│
├── app.py
├── requirements.txt
├── chats.json
├── uploads/
├── assets/
└── README.md
```

---

# ⚙ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/CheboluGayatri/CodeGenAiand_Explainer.git

cd CodeGenAiand_Explainer
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Install Tesseract OCR

### Windows

Download and install:

https://github.com/UB-Mannheim/tesseract/wiki

### Ubuntu / Linux

```bash
sudo apt update
sudo apt install tesseract-ocr
```

### macOS

```bash
brew install tesseract
```

---

## 4. Install Ollama

Download Ollama:

https://ollama.com

Pull the required model:

```bash
ollama pull gemma:2b
```

Start the Ollama server:

```bash
ollama serve
```

---

# ▶ Running the Application

Launch the Streamlit application:

```bash
streamlit run app.py
```

Open your browser and visit:

```
http://localhost:8501
```

---

# 📂 Supported File Types

| File Type | Processing Method |
|-----------|-------------------|
| PNG | OCR |
| JPG | OCR |
| JPEG | OCR |
| BMP | OCR |
| TIFF | OCR |
| PDF | pdfplumber |
| DOCX | python-docx |
| TXT | Native Reader |
| CSV | Pandas |

---

# 💡 Use Cases

- AI-assisted code generation
- Code explanation and learning
- Bug detection and debugging
- Code review
- Code optimization
- OCR-based code extraction
- Programming education
- Developer productivity

---

# 🚀 Future Enhancements

- Support multiple LLMs
- Voice-based prompts
- Multi-language code generation
- Secure code execution sandbox
- Export AI responses
- User authentication
- Cloud AI model integration

---

# 👩‍💻 Internship Details

**Organization:** Infosys Springboard

**Program:** Internship 6.0 (B2)

**Project:** CodeGenie – AI Explainer & Code Generator

**Domain:** Artificial Intelligence

**Duration:** September 2025 – November 2025

---

# 👩‍💻 Developer

**Gayatri Chebolu**

AI • Machine Learning • Generative AI Enthusiast

GitHub: https://github.com/CheboluGayatri

---

# ⭐ Support

If you found this project helpful, consider giving the repository a **⭐ Star** on GitHub. Your support encourages continued development and future improvements.
