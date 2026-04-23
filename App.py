from faster_whisper import WhisperModel
from groq import Groq
from gtts import gTTS
import gradio as gr

# =========================
# API KEY (PUT YOUR KEY)
# =========================
GROQ_API_KEY = "Groq_Api_Key"

client = Groq(api_key=GROQ_API_KEY)

# =========================
# MODELS
# =========================
model = WhisperModel("tiny")

# =========================
# VOICE → TEXT
# =========================
def transcribe(audio_path):
    try:
        segments, _ = model.transcribe(audio_path)
        return " ".join([seg.text for seg in segments]).strip()
    except:
        return ""

# =========================
# AI RESPONSE
# =========================
def generate_response(text):
    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are VoxMind AI, a smart voice assistant. "
                        "Explain answers clearly, step-by-step, especially for technical topics like AI, ML, and programming."
                    )
                },
                {"role": "user", "content": text}
            ],
            model="llama-3.1-8b-instant"
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Error: {str(e)}"

# =========================
# TEXT → VOICE
# =========================
def text_to_speech(text):
    file = "response.mp3"
    tts = gTTS(text)
    tts.save(file)
    return file

# =========================
# PIPELINE
# =========================
def voice_to_voice(audio):
    if audio is None:
        return "No audio detected", "Try again", None

    audio_path = audio
    if isinstance(audio, tuple):
        audio_path = audio[1]

    user_text = transcribe(audio_path)

    if not user_text:
        return "Could not understand audio", "Speak clearly", None

    ai_text = generate_response(user_text)
    ai_voice = text_to_speech(ai_text)

    return user_text, ai_text, ai_voice

# =========================
# UI
# =========================
app = gr.Interface(
    fn=voice_to_voice,
    inputs=gr.Audio(type="filepath", label="🎤 Speak or Upload Audio"),
    outputs=[
        gr.Textbox(label="📝 You Said", lines=5),
        gr.Textbox(label="🤖 VoxMind AI Response", lines=8),
        gr.Audio(label="🔊 AI Voice")
    ],
    title="🎙️ VoxMind AI - Voice Assistant",
    description="Speak → AI understands → AI responds with voice"
)

app.launch()
