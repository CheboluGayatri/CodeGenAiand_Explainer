import streamlit as st
import uuid, datetime, json, os, subprocess, socket
from PIL import Image, ImageOps, ImageFilter
import io
import pandas as pd
import time # Import time for potential delay

# Optional libs (graceful fallback)
try:
    import pytesseract
    # CRITICAL: If you are on Windows and Tesseract is NOT in your system PATH, 
    # you MUST UNCOMMENT and EDIT the line below to point to tesseract.exe.
    # import pytesseract 
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe" 
except Exception:
    pytesseract = None

try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    from docx import Document
except Exception:
    Document = None

# Ollama (optional)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

# Page config
st.set_page_config(page_title="CodeGenAI and Explainer", layout="wide")

# --- Paths / storage
DATA_FILE = "chats.json"
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# System instruction for model
SYSTEM_INSTRUCTION = (
    "You are CodeGenesis, an expert AI assistant specialized in generating, explaining, and correcting code. "
    "When given a file or code snippet, analyze it for syntax, logic, and security issues. Provide a corrected complete code block "
    "and a detailed explanation of all fixes. Keep answers concise and focused. Use markdown code fences for code."
)

# ---------- Helpers ----------
def now_iso(): return datetime.datetime.now().isoformat()
def make_chat(title=None):
    return {"id": uuid.uuid4().hex, "title": title or "New Chat", "messages": [], "created_at": now_iso()}

def load_chats():
    if os.path.exists(DATA_FILE):
        try:
            return json.load(open(DATA_FILE, "r", encoding="utf-8"))
        except Exception:
            return []
    return []

def save_chats(chats):
    open(DATA_FILE, "w", encoding="utf-8").write(json.dumps(chats, indent=2, ensure_ascii=False))

def ensure_ollama_running():
    if not OLLAMA_AVAILABLE:
        return False
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=0.2):
            return True
    except Exception:
        return False

def get_installed_models():
    if not OLLAMA_AVAILABLE:
        return ["(Ollama not installed)"]
    if not ensure_ollama_running():
        return ["(Ollama server down)"]
    try:
        # Check for model list, suppress errors
        res = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=False)
        lines = [l.strip().split()[0] for l in res.stdout.splitlines()[1:] if l.strip()]
        return lines or ["gemma:2b"]
    except Exception:
        return ["(Error listing models)"]

# ---------- Robust OCR / extraction ----------
def preprocess_image_for_ocr(pil_img):
    """
    Convert to grayscale, optionally resize, apply a mild filter and threshold to improve OCR.
    Returns a PIL image ready for pytesseract.
    """
    try:
        # convert to grayscale
        img = pil_img.convert("L")
        # increase contrast a bit
        img = ImageOps.autocontrast(img)
        # optionally resize small images to improve OCR
        w, h = img.size
        if max(w, h) < 1200:
            scale = max(1, int(1200 / max(w, h)))
            img = img.resize((w * scale, h * scale), Image.LANCZOS)
        # mild median filter to reduce noise
        img = img.filter(ImageFilter.MedianFilter(size=3))
        # apply simple binary threshold
        img = img.point(lambda p: 255 if p > 180 else 0)
        return img
    except Exception as e:
        # If preprocessing fails, return the original image to at least try OCR
        return pil_img

def extract_text(path):
    """
    Returns extracted text for supported file paths.
    For images, runs OCR (pytesseract). For PDF uses pdfplumber. For docx uses python-docx.
    """
    ext = path.split(".")[-1].lower()
    try:
        if ext in ["jpg", "jpeg", "png", "bmp", "tiff"]:
            if not pytesseract:
                return "(ERROR: pytesseract not installed. Install with `pip install pytesseract` and ensure Tesseract OCR is installed.)"
            try:
                with Image.open(path) as im:
                    im = preprocess_image_for_ocr(im)
                    # Use image_to_string, which handles the Tesseract call
                    text = pytesseract.image_to_string(im)
                    return text.strip() or "(No text detected in image.)"
            except pytesseract.TesseractNotFoundError:
                return "(ERROR: Tesseract executable not found. Check if Tesseract is installed and its path is correctly set in the Python code.)"
            except subprocess.CalledProcessError as e:
                return f"(ERROR: Tesseract process failed with exit code {e.returncode}. This often indicates bad permissions or configuration.)"
            except Exception as e:
                return f"(OCR failed: {type(e).__name__}: {e}. Check Tesseract path setting.)"

        if ext == "pdf":
            if not pdfplumber:
                return "(ERROR: pdfplumber not installed. Install with `pip install pdfplumber`.)"
            try:
                txt = ""
                with pdfplumber.open(path) as pdf:
                    for p in pdf.pages:
                        txt += (p.extract_text() or "") + "\n"
                return txt.strip() or "(No text detected in PDF.)"
            except Exception as e:
                return f"(PDF read failed: {type(e).__name__}: {e})"

        if ext == "docx":
            if not Document:
                return "(ERROR: python-docx not installed. Install with `pip install python-docx`.)"
            try:
                doc = Document(path)
                txt = "\n".join(p.text for p in doc.paragraphs)
                return txt.strip() or "(Empty DOCX file.)"
            except Exception as e:
                return f"(DOCX read failed: {type(e).__name__}: {e})"

        if ext == "txt":
            return open(path, "r", encoding="utf-8", errors="ignore").read()

        if ext == "csv":
            try:
                return pd.read_csv(path, nrows=200).to_string()
            except Exception as e:
                return f"(CSV read failed: {type(e).__name__}: {e})"

        return "(Unsupported file type.)"
    except Exception as e:
        return f"(Extraction error: {type(e).__name__}: {e})"

# ---------- Initialize session state ----------
if "chats" not in st.session_state:
    st.session_state["chats"] = load_chats()

# Ensure we have at least one clean chat
if not st.session_state["chats"]:
    st.session_state["chats"].append(make_chat("New Chat"))
    save_chats(st.session_state["chats"])
elif st.session_state["chats"][0]["title"] == "Welcome":
    st.session_state["chats"][0]["title"] = "New Chat"
    st.session_state["chats"][0]["messages"] = []
    save_chats(st.session_state["chats"])


if "current_chat" not in st.session_state:
    st.session_state["current_chat"] = st.session_state["chats"][0]["id"]
if "show_upload" not in st.session_state:
    st.session_state["show_upload"] = False
if "llm_running" not in st.session_state:
    st.session_state["llm_running"] = False
if "new_message_to_process" not in st.session_state:
    st.session_state["new_message_to_process"] = False
if "message_content" not in st.session_state:
    st.session_state["message_content"] = ""
# pending file store
if "pending_file_upload" not in st.session_state:
    st.session_state["pending_file_upload"] = None

# local refs
chats = st.session_state["chats"]
chat = next((c for c in chats if c["id"] == st.session_state["current_chat"]), None)
if chat is None:
    chat = make_chat()
    st.session_state["chats"].insert(0, chat)
    st.session_state["current_chat"] = chat["id"]
    save_chats(st.session_state["chats"])

# ---------- Callbacks ----------
def handle_send_click():
    if st.session_state.message_content.strip():
        msg = st.session_state.message_content.strip()
        if chat["title"] == "New Chat":
            chat["title"] = msg[:30] + ("..." if len(msg) > 30 else "")
        # Add visible user message
        chat["messages"].append({"role": "user", "content": msg})
        # Add hidden prompt for LLM
        chat["messages"].append({"role": "user", "content": msg, "hidden": True})
        st.session_state.message_content = ""
        st.session_state.new_message_to_process = True
        save_chats(st.session_state["chats"])

def handle_file_upload(uploaded_file):
    """
    Save uploaded file, extract text (OCR if image), create visible upload message (with image preview for images),
    add hidden prompt for the LLM, then trigger generation.
    """
    if uploaded_file is None:
        return

    # Save file to disk
    filename = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    path = os.path.join(UPLOAD_DIR, filename)
    with open(path, "wb") as f:
        f.write(uploaded_file.getvalue())

    # Extract text
    extracted = extract_text(path)
    
    # 1. Add visible messages (Image preview + OCR status)
    if uploaded_file.type.startswith("image"):
        chat["messages"].append({"role": "user", "content": f"📷 Uploaded image: **{uploaded_file.name}**.", "image_path": path})
        # Use extracted text to show status/error
        notice_content = ""
        if extracted.startswith("(ERROR"):
            notice_content = f"**❌ OCR FAILED:** {extracted}"
        elif "No text detected" in extracted:
            notice_content = "⚠️ **Extraction Warning:** No code text detected in the image. Trying to proceed anyway."
        else:
            notice_content = "thinking....."
            
        chat["messages"].append({"role": "assistant", "content": notice_content})
    else:
        chat["messages"].append({"role": "user", "content": f"📎 Uploaded file: **{uploaded_file.name}**."})

    # 2. Title the chat
    if chat["title"] == "New Chat":
        chat["title"] = uploaded_file.name.replace(".", "_")[:30]

    # 3. Build LLM prompt: instructions + extracted text
    explicit_instruction = (
        "**CRITICAL:** Analyze the extracted content below for programming errors (syntax, logic, security). "
        "Provide a concise summary, then the corrected complete code block and a detailed explanation of the fixes. "
        "If the extracted content contains an OCR error message, report the extraction failure and explain why you cannot proceed."
    )
    file_type_tag = "(Image/OCR)" if uploaded_file.type.startswith("image") else "(Document)"
    
    # Send the raw extracted content, even if it's an error message
    llm_prompt = f"File: {uploaded_file.name} {file_type_tag}\n{explicit_instruction}\n\n---\n{extracted[:12000]}"

    # 4. Add the hidden prompt for the LLM to process
    chat["messages"].append({"role": "user", "content": llm_prompt, "hidden": True})

    # Clear upload UI and trigger generation
    st.session_state["show_upload"] = False
    st.session_state["pending_file_upload"] = None
    st.session_state["new_message_to_process"] = True
    save_chats(chats)
    time.sleep(0.1) # Add a small delay
    st.rerun()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("CodeGenesis")
    if OLLAMA_AVAILABLE:
        if not ensure_ollama_running():
            st.error("❌ Ollama server not reachable (run `ollama serve`).")
    else:
        st.info("ℹ️ Ollama library not installed — assistant will show a fallback response.")

    st.subheader("Model Selection")
    models = get_installed_models()
    model = st.selectbox("Model", models, index=0, key="selected_model")

    if st.button("➕ New Chat", width='stretch'):
        nc = make_chat()
        st.session_state["chats"].insert(0, nc)
        st.session_state["current_chat"] = nc["id"]
        save_chats(st.session_state["chats"])

    st.subheader("Chat History")
    search = st.text_input("🔍 Search", key="sidebar_search")
    chats_list = [c for c in st.session_state["chats"] if search.lower() in c["title"].lower()] if search else st.session_state["chats"]

    for c in chats_list:
        cols = st.columns([0.8, 0.2])
        with cols[0]:
            is_current = c["id"] == st.session_state["current_chat"]
            if st.button(c["title"], key=f"chat-{c['id']}", width='stretch'):
                st.session_state["current_chat"] = c["id"]
        with cols[1]:
            if st.button("🗑", key=f"del-{c['id']}", width='stretch'):
                st.session_state["chats"] = [x for x in st.session_state["chats"] if x["id"] != c["id"]]
                save_chats(st.session_state["chats"])
                if st.session_state["chats"]:
                    st.session_state["current_chat"] = st.session_state["chats"][0]["id"]

# ---------------- MAIN CHAT DISPLAY ----------------
st.markdown("""
<style>
.stApp > header { display: none; }
.main { padding-bottom: 160px; }
.fixed-input-container {
    position: fixed; bottom: 0; left: 0; right: 0; z-index: 1000;
    background: #f0f2f6; border-top: 1px solid #ccc; padding: 12px;
    width: calc(100% - 300px);
}
</style>
""", unsafe_allow_html=True)

for msg in chat["messages"]:
    if msg.get("hidden"):
        continue
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("image_path"):
            try:
                st.image(msg["image_path"], caption=os.path.basename(msg["image_path"]), width='stretch')
            except Exception:
                st.write("*(Could not display image)*")

# ---------------- LLM GENERATION ----------------
if st.session_state["new_message_to_process"] and not st.session_state["llm_running"]:
    st.session_state["llm_running"] = True
    st.session_state["new_message_to_process"] = False

    with st.chat_message("assistant"):
        placeholder = st.empty()
        partial = ""

        # Gather last hidden user message as final prompt
        try:
            final_prompt = next((m["content"] for m in reversed(chat["messages"]) if m.get("hidden") and m["role"] == "user"), None)
            if final_prompt is None:
                final_prompt = chat["messages"][-1]["content"] if chat["messages"] else "Analyze nothing."
        except Exception:
            final_prompt = "Analyze nothing."

        if not OLLAMA_AVAILABLE or not ensure_ollama_running():
            # Fallback for when Ollama is not running/installed
            fallback_msg = (
                "⚠️ Local LLM (Ollama) not available. To enable automatic code-fix outputs, install and run Ollama.\n\n"
            )
            placeholder.markdown(fallback_msg)
            
            extracted_preview = final_prompt
            if len(extracted_preview) > 2000:
                extracted_preview = extracted_preview[:2000] + "\n\n...(Content Truncated)"

            # Show extraction/prompt for debugging
            if "File: " in extracted_preview:
                placeholder.markdown(f"**Preview of prompt sent to LLM (first 2000 chars):**\n\n```\n{extracted_preview}\n```\n")
                
            assistant_content = fallback_msg + "Please check the Ollama server status. The full prompt content was captured and is ready for the LLM when it runs."
            
            chat["messages"].append({"role": "assistant", "content": assistant_content})
            save_chats(chats)
        else:
            # Use Ollama streaming API
            try:
                messages_for_ollama = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
                # Only include the system instruction and the final prompt
                messages_for_ollama.append({"role": "user", "content": final_prompt})

                # Stream response into placeholder
                for chunk in ollama.chat(model=model, messages=messages_for_ollama, stream=True):
                    content = chunk.get("message", {}).get("content", "")
                    if content:
                        partial += content
                        placeholder.markdown(partial)
                # finished
                chat["messages"].append({"role": "assistant", "content": partial})
                save_chats(chats)
            except Exception as e:
                err = f"❌ Error during LLM generation: {type(e).__name__}: {e}. Check if the selected model `{model}` is installed via `ollama pull {model}`."
                placeholder.markdown(err)
                chat["messages"].append({"role": "assistant", "content": err})
                save_chats(chats)

    st.session_state["llm_running"] = False

# ---------------- INPUT BAR / UPLOAD ----------------
uploaded_file_placeholder = None
with st.container():
    st.markdown('<div class="fixed-input-container">', unsafe_allow_html=True)
    
    # 🌟 CRITICAL FIX: The logic here was slightly misplaced, preventing the uploader from showing.
    # It is now placed outside the columns but inside the container.
    if st.session_state["show_upload"]:
        with st.expander("Upload file for analysis", expanded=True):
            # File uploader with a fixed key
            uploaded_file_placeholder = st.file_uploader(
                "Upload file (image/pdf/docx/txt/csv)",
                type=["jpg", "jpeg", "png", "pdf", "docx", "txt", "csv"],
                key="file_upload_widget",
                label_visibility="collapsed"
            )

    # Persist uploaded file to session_state for reliable processing across reruns
    if uploaded_file_placeholder is not None:
        st.session_state["pending_file_upload"] = uploaded_file_placeholder

    # Process pending file if present and not currently running LLM
    if st.session_state.get("pending_file_upload") is not None and not st.session_state["llm_running"]:
        # Give spinner and then handle upload
        with st.spinner("Processing uploaded file (OCR / extract)..."):
            handle_file_upload(st.session_state["pending_file_upload"])
            # Clear the widget immediately after processing the file
            st.session_state["file_upload_widget"] = None 
            st.rerun() # Re-run one last time to clear the widget UI

    # Input area (message)
    col_plus, col_text = st.columns([0.07, 0.93], gap="small")
    with col_plus:
        if st.button("➕", key="upload_toggle_btn", disabled=st.session_state["llm_running"], width='stretch'):
            st.session_state["show_upload"] = not st.session_state["show_upload"]
            time.sleep(0.05) # Add small delay to ensure UI update
            st.rerun()
    with col_text:
        st.text_input(
            "Type message here...",
            key="message_content",
            label_visibility="collapsed",
            disabled=st.session_state["llm_running"],
            value=st.session_state["message_content"],
            on_change=handle_send_click,
            placeholder="Type your coding question, or upload an image/file containing code..."
        )
    st.markdown('</div>', unsafe_allow_html=True)