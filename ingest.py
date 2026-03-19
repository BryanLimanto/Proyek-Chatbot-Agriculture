import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Menghilangkan log info/warning TF
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Mematikan pesan oneDNN
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions

# --- KONFIGURASI ---
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "agri2_knowledge_base"
DATA_PATH = "./dataset/" # Folder berisi 8 PDF Anda

# 1. Inisialisasi Nomic Embedder
# Nomic-embed-text-v1.5 sangat kuat untuk teks teknis dan multibahasa
nomic_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)

def start_ingestion():
    # 2. Inisialisasi ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, 
        embedding_function=nomic_ef,
        metadata={"hnsw:space": "cosine"}
    )

    # 3. Load & Split Dokumen
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800, # Ukuran chunk optimal untuk Nomic
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Memproses: {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            pages = loader.load()
            chunks = text_splitter.split_documents(pages)

            # 4. Masukkan ke Database
            documents = [c.page_content for c in chunks]
            metadatas = [{"source": file, "page": c.metadata['page']} for c in chunks]
            ids = [f"{file}_{i}" for i in range(len(chunks))]

            # TAMBAHKAN VALIDASI INI
            if documents:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                print(f"✅ Berhasil memasukkan {len(documents)} chunks dari {file}")
            else:
                print(f"⚠️ Peringatan: Tidak ada teks yang diekstrak dari {file}. Melewati...")

if __name__ == "__main__":
    start_ingestion()