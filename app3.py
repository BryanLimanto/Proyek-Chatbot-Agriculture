import streamlit as st
from PIL import Image
import io
from gtts import gTTS
from streamlit_mic_recorder import mic_recorder
from dotenv import load_dotenv
import os
from backend3 import KnowledgeBaseChatbot, PlantDiseaseClassifier

# KONFIGURASI HALAMAN
st.set_page_config(page_title="AgriBot AI Voice", page_icon="🌱", layout="wide")

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


@st.cache_resource(show_spinner="Menyiapkan Sistem...")
def load_system():
    bot = KnowledgeBaseChatbot(
        chroma_dir="./chroma_db",
        collection_name="agri2_knowledge_base", 
        groq_api_key=GROQ_API_KEY
    )
    vision_model = None
    try:
        vision_model = PlantDiseaseClassifier(
            model_path="./model/model_kentang_resnet.tflite",
            labels_path="./model/labels.txt"
        )
    except:
        st.warning("Sistem Vision maintenance.")
    return bot, vision_model

chatbot, vision_model = load_system()

# Helper: Text to Speech
def speak_text(text):
    # Bersihkan teks dari markdown agar suara lebih natural
    clean_text = text.replace("*", "").replace("#", "")
    tts = gTTS(text=clean_text, lang='id')
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    return audio_fp

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🌱 AgriBot Multi-Modal (Voice & Vision)")

tab_chat, tab_vision = st.tabs(["💬 Konsultasi Teks & Suara", "📸 Identifikasi Foto"])

final_query = None
image_to_process = None

with tab_chat:
    col1, col2 = st.columns([4, 1])
    
    with col2:
        st.write("🎙️ Rekam Suara:")
        audio_input = mic_recorder(
            start_prompt="Mulai Rekam",
            stop_prompt="Kirim Suara",
            key='voice_rec'
        )

    with col1:
        text_input = st.chat_input("Tanyakan sesuatu...")
        
    # Logika Input Suara
    if audio_input:
        with st.spinner("Mengonversi suara ke teks..."):
            transcript = chatbot.transcribe_audio(audio_input['bytes'])
            if transcript:
                final_query = transcript
                st.success(f"Dideteksi: '{transcript}'")

    # Logika Input Teks
    if text_input:
        final_query = text_input

with tab_vision:
    uploaded_file = st.file_uploader("Unggah foto daun", type=["jpg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width=300)
        if st.button("Analisis Gambar"):
            if vision_model:
                prediction = vision_model.predict(img)
                label = prediction['class_name']
                final_query = f"Tanaman saya terdeteksi {label}. Bagaimana cara menanganinya?"
                image_to_process = img

# PROSES CHAT
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Tombol play suara untuk setiap jawaban bot
            if st.button("🔊 Putar Suara", key=f"play_{msg['content'][:10]}"):
                audio_data = speak_text(msg["content"])
                st.audio(audio_data, format='audio/mp3', autoplay=True)

if final_query:
    with st.chat_message("user"):
        st.markdown(final_query)
    
    st.session_state.messages.append({"role": "user", "content": final_query})

    with st.spinner("AgriBot sedang memproses..."):
        result = chatbot.chat(query=final_query)
        response_text = result['response']

        with st.chat_message("assistant"):
            st.markdown(response_text)
            
            # Tampilkan Audio Otomatis untuk jawaban baru
            audio_out = speak_text(response_text)
            st.audio(audio_out, format='audio/mp3')
            
            if result.get('search_results'):
                with st.expander("📚 Referensi Dokumen"):
                    for res in result['search_results']:
                        st.caption(f"{res['chunk_source']} ({res['similarity_percent']:.1f}%)")

        st.session_state.messages.append({"role": "assistant", "content": response_text})