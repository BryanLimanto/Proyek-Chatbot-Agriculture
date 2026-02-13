import streamlit as st
import time
from PIL import Image
# Import class dari backend baru
from backend import KnowledgeBaseChatbot, PlantDiseaseClassifier
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Agri-Chatbot AI (Groq)",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Asisten Cerdas Pertanian (Groq Powered)")
st.markdown("Sistem Multimodal: Deteksi penyakit tanaman & Konsultasi Cepat.")


# ==========================================
# INISIALISASI SISTEM (CACHED)
# ==========================================
@st.cache_resource(show_spinner="Sedang menghubungkan ke Groq Cloud & Vision Model...")
def load_system():
    # 1. Konfigurasi Chatbot
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "knowledge_base_new2"
    
    bot = KnowledgeBaseChatbot(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        groq_api_key=api_key, # Pass API Key di sini
        cross_encoder_model="BAAI/bge-reranker-base"
    )

    # 2. Konfigurasi Vision Model (TFLite)
    vision_model = None
    try:
        vision_model = PlantDiseaseClassifier(
            model_path="./model/model_kentang_resnet.tflite",
            labels_path="./model/labels.txt"
        )
        print("✅ Vision Model berhasil dimuat.")
    except Exception as e:
        print(f"⚠️ Warning: Vision model gagal dimuat. ({e})")

    return bot, vision_model

# Load Sistem
try:
    chatbot, vision_model = load_system()
    st.success("Sistem Terhubung! 🚀", icon="✅")
    time.sleep(1)
    st.empty()
except Exception as e:
    st.error(f"Critical Error: {e}")
    st.stop()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    st.info(f"Engine: Groq Llama 3 70B") # Indikator model
    
    st.divider()
    if st.button("Bersihkan Chat", type="primary"):
        st.session_state.messages = []
        st.rerun()

# ==========================================
# LOGIKA UTAMA (TABS)
# ==========================================
tab_chat, tab_vision = st.tabs(["💬 Chat & Konsultasi", "📸 Analisis Foto Daun"])

final_user_query = None
processed_image = None

# --- TAB 1: INPUT TEXT MANUAL ---
with tab_chat:
    text_input = st.chat_input("Ketik pertanyaan Anda...")
    if text_input:
        final_user_query = text_input

# --- TAB 2: INPUT FOTO (VISION) ---
with tab_vision:
    uploaded_file = st.file_uploader("Upload Foto (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption='Foto Daun', use_container_width=True)
        
        with col2:
            if st.button("🔍 Analisis Penyakit", type="primary"):
                if vision_model:
                    with st.spinner("Sedang menganalisis tekstur daun..."):
                        prediction = vision_model.predict(image)
                        label = prediction['class_name']
                        conf = prediction['confidence'] * 100
                        
                        if conf > 60:
                            st.success(f"**Hasil Deteksi:** {label}")
                            st.info(f"Keyakinan: {conf:.2f}%")
                            final_user_query = f"Tanaman saya kena {label}. Apa obatnya?"
                            processed_image = image
                        else:
                            st.warning(f"Terdeteksi: {label}, keyakinan rendah ({conf:.2f}%).")
                else:
                    st.error("Model Vision tidak aktif.")

# ==========================================
# PROSES CHATBOT (RAG)
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image_data"):
            st.image(message["image_data"], width=200)
        st.markdown(message["content"])

if final_user_query:
    with st.chat_message("user"):
        if processed_image:
            st.image(processed_image, width=200)
        st.markdown(final_user_query)
    
    st.session_state.messages.append({
        "role": "user", 
        "content": final_user_query,
        "image_data": processed_image
    })

    # PROSES KE BACKEND
    with st.spinner("🤖 Groq AI sedang mengetik..."):
        try:
            result = chatbot.chat(query=final_user_query, n_results=4)
            response_text = result['response']
            
            with st.chat_message("assistant"):
                st.markdown(response_text)
                
                # Tampilkan Sumber jika ada (bukan social chat)
                if result['search_results']:
                    with st.expander("📚 Sumber Referensi"):
                        for res in result['search_results']:
                            st.caption(f"**{res['chunk_source']}** ({res['similarity_percent']:.1f}%)")
                            st.text(res['document'][:150] + "...")

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response_text
            })
            
        except Exception as e:
            st.error(f"Error: {e}")