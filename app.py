import gradio as gr
from llama_cpp import Llama
import os
import glob
import platform
import subprocess
from transformers import AutoConfig
import sys
import time
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# =========================================================
# CONFIGURATION
# =========================================================
MODELS_DIR = os.path.join(os.path.expanduser("~"), "Documents", "models")
model_files = glob.glob(os.path.join(MODELS_DIR, "*.gguf"))

if not model_files:
    model_files = [None]  # Handle the case with no models

llm = None  # Global model instance

# =========================================================
# GPU DETECTION
# =========================================================
def detect_gpu_backend():
    """Detects whether CUDA, Metal, or ROCm is available."""
    system = platform.system().lower()
    
    # --- NVIDIA (CUDA)
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return "cuda"
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass

    # --- AMD (ROCm)
    try:
        subprocess.run(["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return "rocm"
    except FileNotFoundError:
        pass
    except subprocess.CalledProcessError:
        pass

    # --- Apple Silicon (Metal)
    if system == "darwin" and "arm" in platform.machine().lower():
        return "metal"

    return "cpu"

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def get_model_ctx(model_name_or_path):
    """Fetch the model context size (n_ctx) from Hugging Face if available."""
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        if hasattr(config, 'max_position_embeddings'):
            return config.max_position_embeddings
        else:
            return 16384
    except Exception as e:
        print(f"Error fetching context for {model_name_or_path}: {e}")
        return 4096

def load_model(model_path):
    """Loads the selected model with GPU acceleration if available."""
    global llm
    backend = detect_gpu_backend()
    model_ctx = get_model_ctx(model_path)

    try:
        debug_info = f"🔍 Detected backend: {backend.upper()}\n"

        if backend == "cuda":
            debug_info += f"⚙️ Loading model on RTX GPU (Context size: {model_ctx})...\n"
            llm = Llama(
                model_path=model_path,
                n_ctx=model_ctx,
                n_gpu_layers=-1,
                seed=42,
                verbose=False,
                use_mmap=True,
                use_mlock=True,
                fp16=True  # Mixed precision for RTX 3070
            )
            debug_info += "✅ Model loaded on GPU.\n"
        else:
            debug_info += f"⚙️ No GPU detected — loading model on CPU (Context size: {model_ctx})...\n"
            llm = Llama(
                model_path=model_path,
                n_ctx=model_ctx,
                n_gpu_layers=0,
                seed=42,
                verbose=False
            )
            debug_info += "✅ Model loaded on CPU.\n"

        debug_info += f"Model: {os.path.basename(model_path)}\n"
        return debug_info

    except Exception as e:
        llm = None
        return f"❌ Error loading model: {e}"

def generate_text(prompt, history):
    """Generate text with the loaded model."""
    global llm

    if llm is None:
        return "⚠️ No model loaded. Please select and load a model first.", history

    try:
        if history is None:
            history = []

        history.append({"role": "user", "content": prompt})

        output = llm.create_chat_completion(
            messages=history,
            max_tokens=256,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1
        )

        assistant_message = output["choices"][0]["message"]["content"].strip()

        history.append({"role": "assistant", "content": assistant_message})

        return "", history

    except Exception as e:
        return f"❌ Error generating text: {e}", history

def cleanup():
    """Clean up the model resources."""
    global llm
    if llm:
        try:
            llm.close()
        except AttributeError:
            pass
        llm = None

# =========================================================
# GRADIO INTERFACE WITH CUSTOM DARK THEME
# =========================================================
css = """
/* Global styling */
:root {
    --bg: #000000;
    --text: #00FF41;
    --accent: #33FF88;
    --panel: #081018;
    --muted: #050505;
}

/* base styles */
* {
    box-sizing: border-box;
}

html, body {
    height: 100%;
    margin: 0;
    font-family: "Courier New", Courier, monospace;
    color: var(--text);
    background: var(--bg);
}

.gradio-container {
    background: var(--bg) !important;
    color: var(--text) !important;
    padding: 20px !important;
}

/* Ensure all input elements follow the dark theme */
.gradio-container input, 
.gradio-container textarea,
.gradio-container .input,
.gradio-container .textbox, 
.gradio-container .dropdown,
.gradio-container .gradio-input,
.gradio-container .gradio-textbox {
    background: rgba(0, 0, 0, 0.45) !important;
    color: var(--text) !important;
    border: 1px solid rgba(0, 255, 65, 0.08) !important;
    border-radius: 6px !important;
    padding: 8px !important;
}

.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container .input:focus,
.gradio-container .textbox:focus,
.gradio-container .gradio-input:focus,
.gradio-container .gradio-textbox:focus {
    background: rgba(0, 0, 0, 0.55) !important;
    border-color: var(--accent) !important;
    outline: none !important;
}

/* Buttons */
.gradio-container button, 
.gradio-container .button {
    background: #001400 !important;
    color: var(--text) !important;
    border: 1px solid rgba(0, 255, 65, 0.12) !important;
    padding: 8px 12px !important;
    border-radius: 6px !important;
    cursor: pointer;
}

.gradio-container button:hover, 
.gradio-container .button:hover {
    background: rgba(0, 255, 65, 0.08) !important;
    color: #001100 !important;
}

/* Chat messages */
.gradio-container .chat {
    background: transparent !important;
}

.gradio-container .chat .message {
    background: rgba(0, 0, 0, 0.45) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    margin: 6px 0 !important;
    border: 1px solid rgba(0, 255, 65, 0.04) !important;
}

/* user vs assistant styles */
.gradio-container .chat .message.user {
    background: rgba(0, 0, 30, 0.4) !important;
    color: #66d9ff !important;
}

.gradio-container .chat .message.assistant {
    background: rgba(0, 25, 0, 0.5) !important;
    color: #9cff9c !important;
}

/* Links */
.gradio-container a {
    color: var(--accent) !important;
}
"""

# =========================================================
# WATCHDOG SETUP FOR HOT RELOADING
# =========================================================
class ReloadHandler(FileSystemEventHandler):
    def __init__(self, path):
        self.path = os.path.abspath(path)
    def on_modified(self, event):
        if os.path.abspath(event.src_path) == self.path:
            print("\n🔁 Detected file change, restarting app...")
            os.execv(sys.executable, ["python"] + sys.argv)

def watch_file(path):
    event_handler = ReloadHandler(path)
    observer = Observer()
    observer.schedule(event_handler, path=os.path.dirname(path) or ".", recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

watcher_thread = threading.Thread(target=watch_file, args=(__file__,), daemon=True)
watcher_thread.start()

# =========================================================
# LAUNCH GRADIO INTERFACE
# =========================================================
with gr.Blocks(css=css) as iface:
    gr.Markdown("## 🦙 Local LLaMA Chat (Dark Theme)")

    model_selector = gr.Dropdown(
        choices=model_files,
        label="Select Model File",
        value=model_files[0] if model_files else None,
        interactive=True
    )

    load_button = gr.Button("Load Model")
    input_box = gr.Textbox(label="Input Prompt", lines=2, placeholder="Ask something...")

    # Create a Chatbot component to display conversation history
    chat_history = gr.Chatbot(label="Chat", type="messages")

    # Debug box for model loading logs
    debug_box = gr.Textbox(label="Debug Log", lines=6, interactive=False)

    # Send button
    send_button = gr.Button("Send")

    # Function to load model on button click
    load_button.click(fn=load_model, inputs=model_selector, outputs=debug_box)

    # Create interaction: user submits input, model generates a response
    input_box.submit(fn=generate_text, inputs=[input_box, chat_history], outputs=[input_box, chat_history])
    send_button.click(fn=generate_text, inputs=[input_box, chat_history], outputs=[input_box, chat_history])

    gr.Markdown("💡 Tip: Use a `-chat.gguf` or `-instruct.gguf` model for best results.")

# cleanup on exit (optional)
try:
    iface.launch(server_name="192.168.1.43", server_port=4420)  # Launch on desired IP/Port
finally:
    cleanup()
