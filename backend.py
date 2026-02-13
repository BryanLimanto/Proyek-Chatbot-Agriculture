import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple
import logging
import re
import tensorflow as tf
from symspellpy import SymSpell, Verbosity
import numpy as np
from sentence_transformers import CrossEncoder
from PIL import Image, ImageOps
import torch # Masih butuh torch untuk CrossEncoder/Embedding
from groq import Groq # LIBRARY BARU

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite

# ============================================================================
# KONFIGURASI DAN LOGGING
# ============================================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlantDiseaseClassifier:
    def __init__(self, model_path, labels_path):
        try:
            # Menggunakan safe_mode=False seringkali membantu melewati error deserialisasi input layer
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info(f"✅ Vision Model berhasil dimuat dari {model_path}")
        except Exception as e:
            logger.error(f"❌ Gagal memuat model: {e}")
            raise

        if os.path.exists(labels_path):
            with open(labels_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
        else:
            self.labels = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]

    def predict(self, pil_image):
        # Resize sesuai input model saat training
        img = pil_image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        
        # Normalisasi (Sangat penting jika training menggunakan skala 0-1)
        img_array = img_array / 255.0 
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        return {
            'class_name': self.labels[predicted_class_idx],
            'confidence': float(confidence)
        }

class KnowledgeBaseChatbot:
    def __init__(self, chroma_dir: str, collection_name: str, groq_api_key: str, 
                 cross_encoder_model: str = "BAAI/bge-reranker-base"):
        
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.cross_encoder_model = cross_encoder_model
        
        # 1. INISIALISASI GROQ CLIENT (PENGGANTI LOCAL LLM)
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm_model_name = "llama-3.3-70b-versatile" # Model Groq yang sangat bagus (bisa diganti mixtral-8x7b-32768)

        # 2. EMBEDDING FUNCTION (Wajib sama dengan saat ingest data)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        self._init_chroma_client()
        self._init_symspell()
        self._load_cross_encoder()
        
        self.key_terms = set()
        logger.info("Chatbot (Groq Version) berhasil diinisialisasi")

    def _init_chroma_client(self):
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"ChromaDB collection '{self.collection_name}' berhasil diakses")
        except Exception as e:
            logger.error(f"Error inisialisasi ChromaDB: {e}")
            raise

    def _load_cross_encoder(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(self.cross_encoder_model, device=device)
            logger.info(f"Cross-encoder dimuat di {device}")
        except Exception as e:
            logger.error(f"GAGAL memuat Cross-Encoder: {e}")
            self.cross_encoder = None

    def _init_symspell(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        # Load kamus default sederhana
        self._create_default_dictionary()

    def _create_default_dictionary(self):
        # Kamus sederhana untuk fallback
        words = ["kentang", "jagung", "cabai", "penyakit", "daun", "obat", "pupuk", "tanaman"]
        for word in words:
            self.sym_spell.create_dictionary_entry(word, 1)

    def correct_typos(self, text: str) -> str:
        # (Fungsi typo sama seperti sebelumnya, disederhanakan untuk ringkas)
        return text 

    # ============================================================================
    # FUNGSI SIMILARITY SEARCH (SAMA SEPERTI SEBELUMNYA)
    # ============================================================================
    def similarity_search(self, query: str, n_results: int = 5, initial_candidates: int = 20) -> List[Dict[str, Any]]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(initial_candidates, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            if not results['documents'] or not results['documents'][0]:
                return []
            
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            # Reranking Logic
            formatted_results = []
            
            if self.cross_encoder and len(documents) > 1:
                pairs = [(query, doc) for doc in documents]
                scores = self.cross_encoder.predict(pairs)
                
                combined_data = []
                for i, (doc, meta, dist, score) in enumerate(zip(documents, metadatas, distances, scores)):
                    combined_data.append({
                        'document': doc, 'metadata': meta, 'similarity_percent': float(score) * 100, 
                        'chunk_source': meta.get('source', 'Unknown'), 'reranked': True
                    })
                
                # Sort by Cross Encoder Score
                combined_data.sort(key=lambda x: x['similarity_percent'], reverse=True)
                formatted_results = combined_data[:n_results]
            else:
                # Fallback Cosine
                for doc, meta, dist in zip(documents[:n_results], metadatas[:n_results], distances[:n_results]):
                    formatted_results.append({
                        'document': doc, 'metadata': meta, 'similarity_percent': (1-dist)*100,
                        'chunk_source': meta.get('source', 'Unknown'), 'reranked': False
                    })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    # ============================================================================
    # GENERATE RESPONSE (MENGGUNAKAN GROQ)
    # ============================================================================
    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        # 1. Gabungkan Konteks
        context_text = ""
        for res in search_results:
            context_text += f"SUMBER: {res['chunk_source']}\nISI: {res['document']}\n\n"
        
        if not context_text:
            context_text = "Tidak ada informasi spesifik di database."

        # 2. System Prompt
        system_prompt = f"""Anda adalah Asisten Ahli Pertanian. Tugas Anda adalah menjawab pertanyaan petani berdasarkan konteks dokumen yang diberikan.

KONTEKS DOKUMEN:
{context_text}

ATURAN:
1. Jawab HANYA berdasarkan konteks di atas. Jika tidak ada info, katakan tidak tahu.
2. Gunakan Bahasa Indonesia yang ramah dan mudah dipahami petani.
3. Sebutkan nama penyakit dan solusinya dengan format Bullet Points jika ada daftar.
4. Jangan berhalusinasi membuat data sendiri.
"""

        try:
            # 3. Panggil Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                model=self.llm_model_name,
                temperature=0.5,
                max_tokens=1024,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Maaf, terjadi kesalahan pada koneksi AI: {str(e)}"

    def _generate_social_response(self, query: str) -> str:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Anda adalah asisten pertanian yang ramah. Jawab sapaan user dengan singkat (1-2 kalimat)."},
                    {"role": "user", "content": query}
                ],
                model=self.llm_model_name,
                temperature=0.7,
                max_tokens=100,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            return "Halo! Ada yang bisa saya bantu tentang tanaman Anda?"

    def _is_social_query(self, query: str) -> bool:
        social_keywords = ["halo", "hai", "selamat", "pagi", "siang", "sore", "malam", "thanks", "makasih"]
        return any(k in query.lower() for k in social_keywords) and len(query.split()) < 4

    # ============================================================================
    # FUNGSI UTAMA CHAT
    # ============================================================================
    def chat(self, query: str, n_results: int = 3, correct_typos: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        
        # 1. Cek Social Chat
        if self._is_social_query(query):
            return {
                'response': self._generate_social_response(query),
                'search_results': []
            }

        # 2. Retrieval
        search_results = self.similarity_search(query, n_results=n_results)
        
        # 3. Generation (Groq)
        response = self.generate_response(query, search_results)
        
        return {
            'response': response,
            'search_results': search_results
        }

    def clear_memory(self):
        # Tidak terlalu perlu untuk API, tapi bagus untuk CrossEncoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()