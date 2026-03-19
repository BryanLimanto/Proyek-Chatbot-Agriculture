import streamlit as st
import time
from PIL import Image
# Memastikan backend.py berada di direktori yang sama
from dotenv import load_dotenv
from backend import KnowledgeBaseChatbot, PlantDiseaseClassifier

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="AgriBot AI - Kelola Penyakit Tanaman",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan chat
st.markdown("""
    <style>
    .stChatMessage {
        border-radius: 15px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌱 AgriBot: Asisten Spesialis Kentang, Cabai, & Jagung")
st.markdown("Sistem ini menggunakan **Automata Intent Recognition** untuk memahami kebutuhan Anda.")

# ==========================================
# API KEY & CONFIG
# ==========================================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ==========================================
# INISIALISASI SISTEM (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Menyiapkan Otak AgriBot...")
def load_system():
    # Inisialisasi Chatbot RAG
    bot = KnowledgeBaseChatbot(
        chroma_dir="./chroma_db",
        collection_name="agri2_knowledge_base", 
        groq_api_key=api_key
    )

    # Inisialisasi Vision Model
    vision_model = None
    try:
        vision_model = PlantDiseaseClassifier(
            model_path="./model/model_kentang_resnet.tflite",
            labels_path="./model/labels.txt"
        )
    except Exception as e:
        st.warning(f"Sistem Vision sedang maintenance. Mode teks tetap aktif.")

    return bot, vision_model

# Eksekusi Load
chatbot, vision_model = load_system()

# ==========================================
# SIDEBAR & STATE MANAGEMENT
# ==========================================
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Panel Kontrol")
    st.info("Cakupan Tanaman: \n1. Kentang 🥔\n2. Cabai 🌶️\n3. Jagung 🌽")
    
    st.divider()
    if st.button("Hapus Riwayat Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# LOGIKA ANTARMUKA (TABS)
# ==========================================
tab_chat, tab_vision = st.tabs(["💬 Konsultasi Teks", "📸 Identifikasi Foto"])

# Variable untuk menangkap input dari kedua tab
final_query = None
image_to_process = None

# --- TAB 1: KONSULTASI TEKS (Fokus Utama) ---
with tab_chat:
    # Menampilkan pesan sambutan otomatis jika chat kosong
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Halo! Saya AgriBot. Saya bisa membantu menjawab pertanyaan tentang hama/penyakit atau mengidentifikasi foto daun Kentang, Cabai, dan Jagung. Ada yang bisa saya bantu?")

    # Chat Input
    text_input = st.chat_input("Tanyakan sesuatu (misal: 'Coba cek tanaman saya' atau 'Apa itu hama Thrips?')")
    if text_input:
        final_query = text_input

# --- TAB 2: IDENTIFIKASI FOTO (Vision) ---
with tab_vision:
    st.subheader("Analisis Kesehatan Tanaman via Foto")
    uploaded_file = st.file_uploader("Unggah foto daun yang bermasalah", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Foto yang diunggah", width=300)
        
        if st.button("Mulai Analisis Gambar", key="run_vision"):
            if vision_model:
                with st.spinner("Menganalisis pola morfologi daun..."):
                    prediction = vision_model.predict(img)
                    label = prediction['class_name']
                    conf = prediction['confidence'] * 100
                    
                    # Tampilkan hasil di UI tab vision
                    st.success(f"**Prediksi:** {label} ({conf:.1f}%)")
                    
                    # Lempar ke alur chat untuk penjelasan RAG
                    final_query = f"Tanaman saya terdeteksi terkena {label}. Berikan penjelasan dan cara menanganinya."
                    image_to_process = img
            else:
                st.error("Model Vision tidak tersedia.")

# ==========================================
# PEMROSESAN LOGIKA (INTENT & RAG)
# ==========================================

# Tampilkan riwayat chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=150)
        st.markdown(msg["content"])

# Proses input baru (baik dari teks maupun hasil vision)
if final_query:
    # 1. Tampilkan Pesan User
    with st.chat_message("user"):
        if image_to_process:
            st.image(image_to_process, width=150)
        st.markdown(final_query)
    
    st.session_state.messages.append({
        "role": "user", 
        "content": final_query, 
        "image": image_to_process
    })

    # 2. Respon Bot (Menggunakan Intent Recognition dari Backend)
    with st.spinner("AgriBot sedang berpikir..."):
        try:
            # Memanggil fungsi chat di backend.py yang sudah kita modifikasi dengan _get_intent
            result = chatbot.chat(query=final_query)
            
            response_text = result['response']
            intent_type = result['metadata'].get('intent')

            with st.chat_message("assistant"):
                st.markdown(response_text)
                
                # Logic Automata: Jika intent adalah 'COBA', berikan penekanan visual
                if intent_type == "INTENT_COBA":
                    st.warning("Pemberitahuan: Silakan gunakan tombol unggah di Tab 'Identifikasi Foto' untuk memulai proses deteksi gambar.")

                # Tampilkan Referensi jika ada hasil RAG
                if result.get('search_results'):
                    with st.expander("📚 Referensi Dataset PDF"):
                        for res in result['search_results']:
                            st.caption(f"Dokumen: {res['chunk_source']} | Skor: {res['similarity_percent']:.1f}%")
                            st.write(f"_{res['document'][:200]}..._")

            # Simpan Respon ke Session State
            st.session_state.messages.append({"role": "assistant", "content": response_text})

        except Exception as e:
            st.error(f"Maaf, terjadi kendala teknis: {str(e)}")