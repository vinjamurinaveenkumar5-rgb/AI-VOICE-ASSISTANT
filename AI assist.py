import os
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS

# -------------------- Page Config --------------------
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🎙️",
    layout="centered"
)

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(120deg, #0f172a, #1e293b, #111827);
    color: white;
}
.title {
    text-align: center;
    font-size: 45px;
    font-weight: 800;
    color: #38bdf8;
    margin-top: 20px;
    letter-spacing: 1px;
}
.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 18px;
    margin-bottom: 40px;
}
[data-testid="stAudioInput"] {
    background: rgba(255,255,255,0.05);
    border: 2px solid #38bdf8;
    border-radius: 18px;
    padding: 15px;
}
audio {
    width: 100%;
    border-radius: 15px;
    margin-top: 15px;
}
.stMarkdown p {
    font-size: 20px;
    color: #f8fafc;
    line-height: 1.6;
}
h2, h3 {
    color: #38bdf8 !important;
    font-weight: 700;
}
label {
    color: #e0f2fe !important;
    font-size: 18px !important;
    font-weight: 600;
}
.stSuccess {
    background: rgba(34,197,94,0.15);
    border-radius: 12px;
}
.stWarning {
    border-radius: 12px;
}
div:empty {
    display:none;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Load API --------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("GROQ API key is missing. Please set it in your .env file.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Common Whisper hallucinations triggered by background static/silence
HALLUCINATIONS = [
    "thank you", "thanks for watching", "thank you.", "thank you for watching.",
    "subtitles by", "amara.org", "you", "bye", "silence", "thank you very much"
]

# -------------------- AI Response --------------------
def generate_ai_response(transcribed_text):
    # List of candidate models to try in order of preference
    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama3-8b-8192",
        "llama3-70b-8192",
        "mixtral-8x7b-32768"
    ]
    
    for model_name in models_to_try:
        try:
            chat_completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": transcribed_text}
                ],
                max_tokens=500,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            # If a model fails or isn't found, try the next fallback
            continue
            
    raise RuntimeError("None of the specified Groq models are available for your API key.")


# -------------------- Text to Speech --------------------
def convert_text_to_speech(text):
    audio_buffer = BytesIO()
    speech = gTTS(text=text, lang="en")
    speech.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer


# -------------------- UI --------------------
st.markdown('<div class="title">🎙️ AI Voice Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Ask your question using your voice and get an AI-powered spoken response.</div>',
    unsafe_allow_html=True
)

audio_value = st.audio_input("🎤 Record Your Question")

# -------------------- Processing --------------------
if audio_value is not None:

    st.audio(audio_value)

    with st.spinner("🎧 Converting speech to text..."):
        transcription = client.audio.transcriptions.create(
            file=("recording.wav", audio_value.getvalue()),
            model="whisper-large-v3-turbo",
            response_format="json",
            temperature=0.0,
            # Prompt steers Whisper away from inserting filler standard phrases on quiet audio
            prompt="The following is a clear user question spoken into a microphone."
        )

        transcribed_text = transcription.text.strip()

    # Noise and Hallucination Filter
    clean_text = transcribed_text.lower().strip(".")
    is_hallucination = clean_text in HALLUCINATIONS or len(clean_text) < 2

    if transcribed_text and not is_hallucination:

        st.subheader("🗣️ You Asked")
        st.write(transcribed_text)

        with st.spinner("🤖 AI is thinking..."):
            ai_response = generate_ai_response(transcribed_text)

        st.subheader("💡 AI Assistant")
        st.write(ai_response)

        with st.spinner("🔊 Generating voice..."):
            response_audio = convert_text_to_speech(ai_response)

        st.audio(response_audio, format="audio/mp3")
        st.success("✅ Response generated successfully!")

    else:
        st.warning("⚠️ Could not detect clear speech. Please speak closer to the microphone and try again.")