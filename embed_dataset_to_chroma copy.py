import os
import chromadb
import hashlib
import re
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pypdf import PdfReader

# --- KONFIGURASI ---
DATASET_DIR = "./dataset"
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "knowledge_base_new2"

# Ganti ke Model Multilingual yang paham Bahasa Indonesia & Pertanian
# Model ini jauh lebih baik daripada BGE-English untuk kasus Anda
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PDF_MAPPING = {
    "1_penyakit_kentang.pdf": "kentang",
    "2_penyakit_daun_kentang.pdf": "kentang",
    "3_hama_kentang.pdf": "kentang",
    "4_hama_kentang_2.pdf" : "kentang",
    "5_hama_cabai.pdf" : "cabai",
    "6_penyakit_cabai.pdf" : "cabai",
    "7_penyakit_jagung.pdf" : "jagung",
    "8_penyakit_jagung_2.pdf" : "jagung"
}

def normalize_text(text: str) -> str:
    # Bersihkan teks tapi jangan terlalu agresif agar konteks kalimat tidak hilang
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_pdf_text(path: str) -> str:
    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text
    except Exception as e:
        print(f"Error membaca PDF {path}: {e}")
        return ""

def recursive_chunking(text: str, chunk_size=500, overlap=100):
    """
    Teknik chunking yang lebih sederhana tapi efektif untuk RAG.
    Memastikan kalimat tidak terpotong sembarangan.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_len = 0
    
    for sentence in sentences:
        sent_len = len(sentence)
        if current_len + sent_len > chunk_size:
            # Simpan chunk saat ini
            full_chunk = " ".join(current_chunk)
            chunks.append(full_chunk)
            
            # Buat overlap (ambil beberapa kalimat terakhir)
            overlap_len = 0
            new_chunk = []
            for s in reversed(current_chunk):
                if overlap_len + len(s) < overlap:
                    new_chunk.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_chunk = new_chunk
            current_len = overlap_len
            
        current_chunk.append(sentence)
        current_len += sent_len
        
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def main():
    print("🚀 Memulai proses embedding ulang...")
    
    # Inisialisasi Embedding Function Multilingual
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    # Reset Collection agar bersih
    try:
        client.delete_collection(COLLECTION_NAME)
        print("🗑️  Collection lama dihapus.")
    except:
        pass

    # CRUCIAL: Set metadata hnsw:space ke 'cosine'
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"} 
    )

    total_chunks = 0
    
    for filename, category in PDF_MAPPING.items():
        file_path = os.path.join(DATASET_DIR, filename)
        if not os.path.exists(file_path):
            print(f"⚠️  File hilang: {filename}")
            continue
            
        print(f"📄 Memproses: {filename} ({category})")
        raw_text = load_pdf_text(file_path)
        clean_text = normalize_text(raw_text)
        chunks = recursive_chunking(clean_text)
        
        ids = []
        metadatas = []
        docs = []
        
        for i, chunk in enumerate(chunks):
            # ID unik: namafile_index
            chunk_id = f"{filename}_{i}"
            
            ids.append(chunk_id)
            docs.append(chunk)
            metadatas.append({
                "source": filename,
                "category": category,
                "chunk_id": i
            })
            
        if docs:
            collection.add(ids=ids, documents=docs, metadatas=metadatas)
            total_chunks += len(docs)
            print(f"   ✅ {len(docs)} chunks tersimpan.")

    print(f"\n🎉 Selesai! Total {total_chunks} chunks tersimpan di ChromaDB.")
    print("Sekarang database Anda sudah menggunakan Model Multilingual + Cosine Similarity.")

if __name__ == "__main__":
    main()