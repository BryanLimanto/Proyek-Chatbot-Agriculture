import os
import chromadb
import hashlib
import re
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pypdf import PdfReader

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

def normalize_text(text: str) -> str:
    """
    Normalisasi teks dengan:
    1. Lowercase
    2. Hapus whitespace berlebih
    3. (Opsional) Hapus tanda baca
    """
    # Lowercase
    text = text.lower()
    
    # Hapus whitespace berlebih (spasi, tab, newline)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Opsional: Hapus tanda baca (tidak semua kasus perlu)
    # Uncomment jika diperlukan
    # text = re.sub(r'[^\w\s]', '', text)
    
    return text

def load_pdf_text(path: str) -> str:
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text

def chunk_text(text: str):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start = end - CHUNK_OVERLAP
    return chunks

def create_content_fingerprint(text: str) -> str:
    """
    Membuat hash SHA256 dari teks yang sudah dinormalisasi.
    Hash ini akan digunakan sebagai ID yang deterministik.
    """
    normalized_text = normalize_text(text)
    return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()

# PERUBAHAN: Gunakan model multilingual E5
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="intfloat/multilingual-e5-small"
)

# PERUBAHAN: Gunakan PersistentClient, bukan Client
client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.get_or_create_collection(
    name="test1",
    embedding_function=embedding_fn
)

# Untuk track chunk yang sudah diproses (opsional)
processed_chunks = set()

for filename, category in PDF_MAPPING.items():
    pdf_path = os.path.join(DATASET_DIR, filename)

    if not os.path.exists(pdf_path):
        print(f"File tidak ditemukan: {filename}")
        continue

    print(f"Processing {filename} ({category})")

    text = load_pdf_text(pdf_path)
    chunks = chunk_text(text)

    documents = []
    metadatas = []
    ids = []

    for chunk in chunks:
        # Normalisasi teks chunk
        normalized_chunk = normalize_text(chunk)
        
        # Buat content fingerprint
        chunk_id = create_content_fingerprint(chunk)
        
        # Skip jika chunk sudah pernah diproses
        if chunk_id in processed_chunks:
            print(f"  Skipping duplicate chunk: {chunk_id[:16]}...")
            continue
            
        # Untuk versi advanced: merge metadata jika ID sudah ada
        # Di sini kita skip dulu, atau bisa implementasi logika merge
        existing_ids = collection.get(ids=[chunk_id], include=["metadatas"])
        if existing_ids['ids']:
            print(f"  Chunk already exists in DB: {chunk_id[:16]}...")
            # OPSIONAL: Update metadata untuk menambahkan source baru
            # existing_metadata = existing_ids['metadatas'][0]
            # if filename not in existing_metadata.get('sources', []):
            #     # Update logic here
            #     pass
            continue
        
        documents.append(normalized_chunk)
        metadatas.append({
            "kategori": category,
            "source": filename,
            # Opsional: tambahkan field original_length untuk referensi
            "original_length": len(chunk)
        })
        ids.append(chunk_id)
        processed_chunks.add(chunk_id)

    # Tambahkan ke koleksi jika ada dokumen baru
    if documents:
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"  Added {len(documents)} new chunks from {filename}")
    else:
        print(f"  No new chunks from {filename}")

print(f"Embedding selesai. Total unique chunks: {len(processed_chunks)}")