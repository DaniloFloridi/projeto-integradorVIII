import sounddevice as sd
import numpy as np
import tempfile
import scipy.io.wavfile as wavfile
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import queue
import time
import os

SAMPLE_RATE = 16000
CHUNK_SEC = 5   
MODEL_SIZE = "small"        # Modelo que melhor conciliou velocidade com precisão 
DEVICE = "cuda"
COMPUTE_TYPE = "int8"

print(f"Loading Whisper model... {MODEL_SIZE} {DEVICE} {COMPUTE_TYPE}")
whisper = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type=COMPUTE_TYPE
)

def translate_text(text, target_lang="pt"):
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return "[translation failed]"

audio_queue = queue.Queue()
running = False


# Codigo que precisa de atenção --antigo
def record_audio():
    global running
    while running:
        update_status("🎤 Listening... Speak now", "green")
        audio = sd.rec(int(CHUNK_SEC * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
        sd.wait()
        audio_queue.put(audio.squeeze())

def process_audio(target_lang, update_callback):
    global running
    while running:
        if not audio_queue.empty():
            audio_chunk = audio_queue.get()

            update_status("⏳ Processing...", "orange")

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                wavfile.write(f.name, SAMPLE_RATE, audio_chunk)
                wav_path = f.name

            segments, _ = whisper.transcribe(wav_path, beam_size=5)
            os.remove(wav_path)

            text_out = " ".join([seg.text.strip() for seg in segments])
            if text_out.strip():
                translated = translate_text(text_out, target_lang)
                update_callback(text_out, translated)

            update_status("🎤 Listening... Speak now", "green")
        else:
            time.sleep(0.1)

def start_translation():
    global running
    if running:
        return
    running = True
    target_lang = lang_var.get()

    update_status("🎤 Listening... Speak now", "green")
    threading.Thread(target=record_audio, daemon=True).start()
    threading.Thread(target=process_audio, args=(target_lang, update_texts), daemon=True).start()

def stop_translation():
    global running
    running = False
    update_status("⏹ Stopped", "red")

def update_texts(original, translated):
    text_en.configure(state="normal")
    text_trans.configure(state="normal")

    text_en.insert(tk.END, original + "\n")
    text_trans.insert(tk.END, translated + "\n")

    text_en.configure(state="disabled")
    text_trans.configure(state="disabled")

    text_en.see(tk.END)
    text_trans.see(tk.END)

def update_status(message, color):
    status_label.config(text=message, fg=color)

root = tk.Tk()
root.title("🎤 Real-time Speech Translator")
root.geometry("820x620")

lang_var = tk.StringVar(value="pt")
lang_label = tk.Label(root, text="🌍 Target Language:", font=("Arial", 12))
lang_label.pack(pady=5)

lang_dropdown = ttk.Combobox(root, textvariable=lang_var,
                             values=["pt", "es", "fr", "de", "it", "ja", "zh-cn"],
                             state="readonly")
lang_dropdown.pack(pady=5)

status_label = tk.Label(root, text="✅ Ready", font=("Arial", 14, "bold"), fg="blue")
status_label.pack(pady=10)

frame_buttons = tk.Frame(root)
frame_buttons.pack(pady=10)

btn_start = tk.Button(frame_buttons, text="▶️ Start", command=start_translation,
                      bg="green", fg="white", width=12, font=("Arial", 12, "bold"))
btn_start.grid(row=0, column=0, padx=10)

btn_stop = tk.Button(frame_buttons, text="⏹ Stop", command=stop_translation,
                     bg="red", fg="white", width=12, font=("Arial", 12, "bold"))
btn_stop.grid(row=0, column=1, padx=10)

label_en = tk.Label(root, text="📝 Transcribed English", font=("Arial", 12, "bold"))
label_en.pack()
text_en = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD,
                                    state="disabled", font=("Consolas", 11))
text_en.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

label_trans = tk.Label(root, text="🌐 Translation", font=("Arial", 12, "bold"))
label_trans.pack()
text_trans = scrolledtext.ScrolledText(root, height=10, wrap=tk.WORD,
                                       state="disabled", font=("Consolas", 11))
text_trans.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)

root.mainloop()
