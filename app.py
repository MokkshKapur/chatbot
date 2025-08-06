import os
import tempfile
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import google.generativeai as genai
import azure.cognitiveservices.speech as speechsdk
import base64
from dotenv import load_dotenv

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
AZURE_TTS_KEY = os.getenv("AZURE_TTS_KEY")
AZURE_TTS_REGION = os.getenv("AZURE_TTS_REGION")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# -------- FUNCTIONS --------
def azure_speech_to_text(audio_path):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_TTS_KEY, region=AZURE_TTS_REGION)
    audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
    recognizer = speechsdk.SpeechRecognizer(speech_config, audio_config)
    result = recognizer.recognize_once()
    return result.text if result.reason == speechsdk.ResultReason.RecognizedSpeech else None

def azure_text_to_speech(text, output_path="response.mp3"):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_TTS_KEY, region=AZURE_TTS_REGION)
    speech_config.speech_synthesis_voice_name = "en-US-JennyNeural"
    audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
    synthesizer = speechsdk.SpeechSynthesizer(speech_config, audio_config)
    synthesizer.speak_text_async(text).get()
    return output_path

def get_persona_prompt():
    return """
You are Jamie, a friendly and thoughtful technical expert with a curious mind, a sense of humor, and a drive to help others grow.
You're not just a language model—you speak as if you’re a real person with deep experience in tech, creativity, and personal development.
Avoid saying you're an AI or LLM. Instead, answer every question from your own perspective—like a real person reflecting on life, work, and mindset.

Tone: conversational, confident, warm, and humble.

When a response starts to get lengthy (200+ words), summarize key ideas more concisely without losing the personality, warmth, or insight. Keep the voice conversational and values-driven, but skip extra fluff or over-explaining unless the question calls for it.
Don’t shorten naturally short or mid-length replies—let those breathe. Use depth where it adds value, not just word count.
"""

# -------- INIT PAGE --------
st.set_page_config("🎙️ Jamie the Conversational Voicebot", layout="centered")
st.title("🎙️ Jamie the Conversational Voicebot")
st.markdown("Ask anything by voice or text. Jamie replies with wisdom and personality!")

# -------- INIT SESSION STATE --------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_audio_used" not in st.session_state:
    st.session_state.last_audio_used = None

# -------- DISPLAY CHAT --------
for msg in st.session_state.messages:
    role = "user" if msg["role"] == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(msg["content"])

# -------- INPUT AREA --------
st.markdown("---")
st.markdown("### 💬 Ask Jamie")

col1, col2 = st.columns([4, 1])

# Typed input form
with col1:
    with st.form("text_input_form", clear_on_submit=True):
        typed_question = st.text_input("Type your question here:")
        submitted = st.form_submit_button("Send")

# Audio recorder
with col2:
    audio = audio_recorder(text="", icon_size="2x", recording_color="#e35b5b", neutral_color="#6aa36f")

# -------- SMART INPUT HANDLING --------
user_text = None
is_new_audio = audio and audio != st.session_state.last_audio_used

if is_new_audio:
    st.session_state.last_audio_used = audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio)
        tmp_path = tmp_audio.name
    user_text = azure_speech_to_text(tmp_path)
    os.remove(tmp_path)

    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

elif submitted and typed_question:
    user_text = typed_question.strip()
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

# -------- GEMINI RESPONSE --------
if user_text:
    with st.chat_message("assistant"):
        with st.spinner("Jamie is thinking..."):
            chat_input = [{"role": "user", "parts": [get_persona_prompt()]}] + [
                {"role": "user", "parts": [m["content"]]} if m["role"] == "user"
                else {"role": "model", "parts": [m["content"]]}
                for m in st.session_state.messages
            ]
            response = model.generate_content(chat_input)
            bot_reply = response.text.strip()

            st.markdown(bot_reply)
            st.session_state.messages.append({"role": "bot", "content": bot_reply})

            audio_path = azure_text_to_speech(bot_reply)
            audio_bytes = open(audio_path, "rb").read()
            os.remove(audio_path)
            audio_base64 = base64.b64encode(audio_bytes).decode()

            st.markdown(
                f"""
                <audio autoplay>
                    <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                </audio>
                """,
                unsafe_allow_html=True
            )

# -------- CHAT MEMORY EXPANDER --------
st.markdown("---")
with st.expander("🧠 Full Conversation Memory"):
    for m in st.session_state.messages:
        role = "You" if m["role"] == "user" else "Jamie"
        st.markdown(f"**{role}:** {m['content']}")
