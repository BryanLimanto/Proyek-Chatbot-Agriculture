import streamlit as st
import os
from backend2 import KnowledgeBaseChatbot # Pastikan nama file sesuai

# ==========================================
# CONFIG & INITIALIZATION
# ==========================================
st.set_page_config(
    page_title="AI Konsultan Pertanian",
    page_icon="🌱",
    layout="wide"
)

# Inisialisasi Chatbot (Gunakan cache agar model tidak load ulang)
@st.cache_resource
def load_chatbot():
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "knowledge_base_new2"
    MODEL_PATH = "./model/Qwen2.5-3B-Instruct"
    CROSS_ENCODER_MODEL = "unicamp-dl/mmarco-mMiniLM-v6-translated-gpu"
    
    return KnowledgeBaseChatbot(
        chroma_dir=CHROMA_DIR,
        collection_name=COLLECTION_NAME,
        model_path=MODEL_PATH,
        cross_encoder_model=CROSS_ENCODER_MODEL
    )

bot = load_chatbot()

# ==========================================
# SIDEBAR (KONFIGURASI)
# ==========================================
with st.sidebar:
    st.title("⚙️ Pengaturan")
    st.info("Atur parameter pencarian dokumen di sini.")
    
    use_rerank = st.checkbox("Gunakan Cross-Encoder Reranking", value=True)
    correct_typo = st.checkbox("Koreksi Typo Otomatis", value=True)
    num_results = st.slider("Jumlah Referensi Dokumen", min_value=1, max_value=10, value=3)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    # Tampilkan Info Database
    info = bot.get_collection_info()
    st.write(f"📚 **Database:** {info.get('total_documents', 0)} dokumen")
    st.write(f"🎯 **Reranker:** {'Aktif' if info.get('cross_encoder_available') else 'Off'}")

# ==========================================
# MAIN UI
# ==========================================
st.title("🌱 AI Konsultan Pertanian")
st.markdown("Tanyakan solusi penyakit tanaman, hama, atau teknik budidaya berdasarkan database.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Apa keluhan tanaman Anda hari ini?"):
    # 1. Tampilkan pesan user
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Proses Jawaban
    with st.chat_message("assistant"):
        with st.spinner("Menganalisis database dan merumuskan jawaban..."):
            # Jalankan fungsi chat dari class Anda
            result = bot.chat(
                query=prompt,
                n_results=num_results,
                correct_typos=correct_typo,
                use_reranking=use_rerank
            )
            
            response = result['response']
            metadata = result['metadata']
            search_results = result['search_results']

            # Tampilkan Koreksi Typo jika ada
            if metadata.get('typo_correction_applied'):
                st.caption(f"🔧 *Query dikoreksi menjadi: {metadata['corrected_query']}*")

            # Tampilkan Jawaban Utama
            st.markdown(response)

            # 3. Tampilkan Referensi (Expander)
            if search_results:
                with st.expander("📚 Lihat Sumber Referensi"):
                    for i, res in enumerate(search_results):
                        method = "Cross-Encoder" if res.get('reranked') else "Cosine"
                        st.markdown(f"**Poin {i+1}** (Kemiripan: {res['similarity_percent']}% | {method})")
                        st.info(res['document'])
                        st.caption(f"Sumber: {res['chunk_source']} | ID: {res['chunk_id']}")

    # Simpan history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Footer
st.divider()
st.caption("Chatbot ini dikembangkan menggunakan Qwen2.5-3B (4-bit) & RAG System.")