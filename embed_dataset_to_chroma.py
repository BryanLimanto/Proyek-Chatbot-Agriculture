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

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# --- FUNGSI DATA CLEANING OTOMATIS ---

def clean_text_advanced(text: str) -> str:
    """
    Membersihkan noise spesifik dari ekstraksi PDF.
    """
    # 1. Hapus nomor halaman (angka sendirian di awal/akhir baris)
    text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
    
    # 2. Gabungkan kata yang terputus tanda hubung di akhir baris (misal: penya- kit)
    text = re.sub(r'(\w+)-\s*\n(\w+)', r'\1\2', text)
    
    # 3. Hapus karakter non-printable/aneh yang sering muncul di PDF lama
    text = re.sub(r'[^\x00-\x7F]+', ' ', text) 
    
    # 4. Hapus sisa-sisa watermark atau header berulang (Contoh: "Halaman x dari y")
    text = re.sub(r'(?i)halaman\s+\d+\s+dari\s+\d+', '', text)
    
    return text

def normalize_text(text: str) -> str:
    """
    Normalisasi akhir untuk keperluan embedding.
    """
    # Lowercase untuk konsistensi
    text = text.lower()
    
    # Hapus whitespace berlebih (newline, tab jadi satu spasi)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Hapus karakter spesial yang tidak bermakna tapi simpan tanda baca penting (. , ?)
    text = re.sub(r'[^\w\s\.,\?\!]', '', text)
    
    return text

# --- FUNGSI CORE ---

def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    full_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            # Clean per halaman agar noise header/footer hilang lebih efektif
            cleaned_page = clean_text_advanced(page_text)
            full_text.append(cleaned_page)
    
    return "\n".join(full_text)

def chunk_text(text: str):
    # Menggunakan metode sederhana atau bisa ganti ke RecursiveCharacterTextSplitter dari Langchain
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

def create_content_fingerprint(text: str) -> str:
    # Fingerprint harus dibuat dari teks yang SUDAH dinormalisasi
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# --- MAIN EXECUTION ---

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(
    name="agri_knowledge_base",
    embedding_function=embedding_fn
)

processed_chunks = set()

for filename, category in PDF_MAPPING.items():
    pdf_path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(pdf_path):
        print(f"⚠️ File tidak ditemukan: {filename}")
        continue

    print(f"🔍 Processing: {filename} [{category}]")

    raw_text = load_pdf_text(pdf_path)
    chunks = chunk_text(raw_text)

    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        # Step Cleaning & Normalization
        final_chunk = normalize_text(chunk)
        
        # Validasi minimal panjang chunk (buang jika isinya terlalu pendek/kosong)
        if len(final_chunk) < 50:
            continue
            
        chunk_id = create_content_fingerprint(final_chunk)
        
        # Deduplication check
        if chunk_id in processed_chunks:
            continue
            
        # Cek ke DB agar tidak insert ulang jika script dijalankan lagi
        existing = collection.get(ids=[chunk_id])
        if existing['ids']:
            processed_chunks.add(chunk_id)
            continue
        
        documents.append(final_chunk)
        metadatas.append({
            "kategori": category,
            "source": filename
        })
        ids.append(chunk_id)
        processed_chunks.add(chunk_id)

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        print(f" ✅ Berhasil tambah {len(documents)} chunk baru.")
    else:
        print(f" ℹ️ Tidak ada data baru untuk file ini.")

print(f"\n--- SELESAI ---")
print(f"Total Unique Chunks di Database: {collection.count()}")