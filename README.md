# 🚀 CodeGenAI & Explainer

**An AI-powered Code Generation, Explanation & Review Assistant built with Streamlit, Ollama, and OCR.**

🔗 **Live Demo:** https://codegenaiand-explainer.onrender.com

🔗 **GitHub Repository:** https://github.com/CheboluGayatri/CodeGenAiand_Explainer

---

## 📌 Overview

**CodeGenAI & Explainer** is an AI-powered coding assistant that helps developers generate, explain, debug, review, and improve source code through a simple web interface.

The application supports text prompts, document uploads, and image-based code extraction using OCR, making it easy to analyze code from multiple sources. It combines modern AI technologies with an intuitive Streamlit interface to provide a productive coding experience.

This project was developed as part of the **Infosys Springboard Internship 6.0 – CodeGenie: AI Explainer & Code Generator**.

---

# ✨ Features

### 💬 AI Code Assistant

* Generate code from natural language prompts
* Explain complex code line by line
* Fix syntax and logical errors
* Optimize existing code
* Review code quality

### 📄 Document Analysis

Upload and analyze:

* PDF
* DOCX
* TXT
* CSV

The application extracts the content and provides AI-powered explanations and improvements.

### 🖼 OCR-Based Code Extraction

Extract code directly from images using **Tesseract OCR**.

Supported image formats:

* JPG
* JPEG
* PNG
* BMP
* TIFF

Image preprocessing improves OCR accuracy before sending content to the AI model.

### 🤖 Local AI using Ollama

Supports local Large Language Models through Ollama, enabling:

* Offline AI inference
* Faster responses
* Better privacy
* Local processing without sending code to external services

### 📚 Chat History

* Interactive conversations
* Persistent chat sessions
* Easy review of previous AI responses

---

# 🛠 Tech Stack

| Category         | Technologies  |
| ---------------- | ------------- |
| Frontend         | Streamlit     |
| AI Model         | Ollama        |
| OCR              | Tesseract OCR |
| Image Processing | Pillow        |
| PDF Processing   | pdfplumber    |
| DOCX Processing  | python-docx   |
| Data Handling    | Pandas        |
| Language         | Python        |

---

# 📂 Project Structure

```text
CodeGenAiand_Explainer/
│
├── app.py
├── requirements.txt
├── uploads/
├── chats.json
├── README.md
└── assets/
```

---

# ⚙ Installation

## Clone the Repository

```bash
git clone https://github.com/CheboluGayatri/CodeGenAiand_Explainer.git
cd CodeGenAiand_Explainer
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Install Tesseract OCR

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

# Install Ollama

Download from:

https://ollama.com

Pull a model:

```bash
ollama pull gemma:2b
```

Start Ollama:

```bash
ollama serve
```

---

# ▶ Running the Application

```bash
streamlit run app.py
```

Open your browser:

```
http://localhost:8501
```

---

# 🌐 Live Demo

👉 https://codegenaiand-explainer.onrender.com

---

# 📄 Supported File Types

| File Type        | Processing Method |
| ---------------- | ----------------- |
| PNG / JPG / JPEG | OCR               |
| BMP / TIFF       | OCR               |
| PDF              | pdfplumber        |
| DOCX             | python-docx       |
| TXT              | Native Reader     |
| CSV              | Pandas            |

---

# 💡 Use Cases

* AI Code Generation
* Code Explanation
* Debugging
* Code Review
* OCR-based Code Extraction
* Programming Education
* Developer Productivity

---

# 📈 Future Enhancements

* Support multiple LLMs
* Voice input
* Multi-language code generation
* Code execution sandbox
* Export AI responses
* Authentication system
* Cloud model integration

---

# 👩‍💻 Internship Information

**Organization:** Infosys Springboard

**Program:** Internship 6.0 (B2)

**Project:** CodeGenie – AI Explainer & Code Generator

**Domain:** Artificial Intelligence

**Duration:** September 2025 – November 2025

---

# 👩‍💻 Developer

**Gayatri Chebolu**

AI • Machine Learning • Generative AI Enthusiast

GitHub:
https://github.com/CheboluGayatri

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub. Your support helps improve the project and motivates future development.
