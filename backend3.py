import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple, Optional
import logging
import re
import tensorflow as tf
from symspellpy import SymSpell, Verbosity
import numpy as np
from sentence_transformers import CrossEncoder
from PIL import Image, ImageOps
import torch
from groq import Groq
from rank_bm25 import BM25Okapi

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
    """Vision Model untuk klasifikasi penyakit tanaman"""
    
    def __init__(self, model_path: str, labels_path: str):
        try:
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

    def predict(self, pil_image: Image.Image) -> Dict[str, Any]:
        """Prediksi penyakit dari gambar"""
        img = pil_image.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
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
    
    def __init__(
        self, 
        chroma_dir: str, 
        collection_name: str, 
        groq_api_key: str, 
        cross_encoder_model: str = "BAAI/bge-reranker-base",
        min_confidence: float = 0.50, # DINAINAIKKAN AGAR MENGURANGI HALLUCINATION / CONTEXT BLEED
        use_hybrid_search: bool = True
    ):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.cross_encoder_model = cross_encoder_model
        self.min_confidence = min_confidence
        self.use_hybrid_search = use_hybrid_search
        
        # Groq Client
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm_model_name = "llama-3.3-70b-versatile"
        
        # Embedding Function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="nomic-ai/nomic-embed-text-v1.5", 
            trust_remote_code=True
        )
        
        # Initialize components
        self._init_chroma_client()
        self._init_symspell()
        self._load_cross_encoder()
        
        # BM25 untuk hybrid search
        self.bm25 = None
        self.corpus = []
        if self.use_hybrid_search:
            self._init_bm25()
        
        logger.info("✅ Chatbot (Advanced RAG) berhasil diinisialisasi")

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
            logger.info(f"✅ ChromaDB collection '{self.collection_name}' berhasil diakses ({self.collection.count()} documents)")
        except Exception as e:
            logger.error(f"❌ Error inisialisasi ChromaDB: {e}")
            raise

    def _load_cross_encoder(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(self.cross_encoder_model, device=device)
            logger.info(f"✅ Cross-encoder dimuat di {device}")
        except Exception as e:
            logger.warning(f"⚠️ Gagal memuat Cross-Encoder: {e}. Lanjut tanpa reranking.")
            self.cross_encoder = None

    def _init_symspell(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self._create_default_dictionary()

    def _create_default_dictionary(self):
        words = [
            # Istilah Pertanian
            "kentang", "jagung", "cabai", 
            "penyakit", "daun", "batang", "akar", "buah", "bunga",
            "obat", "pupuk", "pestisida", "fungisida", "herbisida",
            "tanaman", "pertanian", "petani", "hama", "virus", "bakteri", "jamur", "hawar",
            # Kata Umum & Sapaan (AGAR TIDAK DI-AUTOCORRECT MENJADI ISTILAH PERTANIAN)
            "halo", "hai", "pagi", "siang", "sore", "malam", "terima", "kasih", "makasih",
            "kabar", "apa", "gimana", "bagaimana", "baik", "sehat", "senang"
            "coba", "identifikasi", "periksa", "cek", "deteksi", "analisis", "siapa", "kamu"
        ]
        for word in words:
            self.sym_spell.create_dictionary_entry(word, 1)

    def _init_bm25(self):
        try:
            all_docs = self.collection.get()
            self.corpus = all_docs['documents']
            tokenized_corpus = [doc.lower().split() for doc in self.corpus]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info(f"✅ BM25 initialized dengan {len(self.corpus)} documents")
        except Exception as e:
            logger.warning(f"⚠️ BM25 init gagal: {e}. Lanjut tanpa hybrid search.")
            self.use_hybrid_search = False

    def correct_typos(self, text: str) -> str:
        words = text.split()
        corrected = []
        for word in words:
            suggestions = self.sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                corrected.append(suggestions[0].term)
            else:
                corrected.append(word)
        return ' '.join(corrected)

    # ========================================================================
    # RETRIEVAL METHODS
    # ========================================================================

    def expand_query(self, original_query: str) -> List[str]:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": """Anda adalah ahli pertanian. Buatlah 3 variasi pertanyaan untuk mendapatkan informasi yang lebih lengkap dari database.

ATURAN:
1. Fokus pada aspek berbeda: penyebab, gejala, solusi, pencegahan
2. Gunakan sinonim dan istilah teknis pertanian
3. Format: satu baris per variasi, memakai numbering atau bullet
4. Tetap dalam konteks pertanian/penyakit tanaman
5. PENTING: Jika ada istilah penyakit lokal (misal: hawar daun, bulai, bercak), sertakan juga nama bahasa Inggrisnya (blight, downy mildew) atau nama ilmiah/latinnya dalam variasi query.

Contoh Input: "hawar daun pada kentang"
Output:
Bagaimana cara mengatasi Phytophthora infestans pada kentang?
Gejala dan solusi untuk potato late blight atau early blight.
Fungisida untuk mengobati penyakit hawar daun kentang."""
                    },
                    {"role": "user", "content": f"Query: {original_query}"}
                ],
                model=self.llm_model_name,
                temperature=0.2,
                max_tokens=512,
            )
            
            variations = chat_completion.choices[0].message.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip() and len(v.strip()) > 10]
            
            all_queries = [original_query] + variations[:3]
            logger.info(f"🔍 Query expansion: {len(all_queries)} variations")
            return all_queries
            
        except Exception as e:
            logger.error(f"❌ Query expansion gagal: {e}")
            return [original_query]

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
            
            formatted_results = []
            
            if self.cross_encoder and len(documents) > 1:
                pairs = [(query, doc) for doc in documents]
                scores = self.cross_encoder.predict(pairs)
                
                for doc, meta, score in zip(documents, metadatas, scores):
                    formatted_results.append({
                        'document': doc,
                        'metadata': meta,
                        'similarity_percent': float(score) * 100,
                        'chunk_source': meta.get('source', 'Unknown'),
                        'reranked': True
                    })
                
                formatted_results.sort(key=lambda x: x['similarity_percent'], reverse=True)
            else:
                for doc, meta, dist in zip(documents, metadatas, distances):
                    formatted_results.append({
                        'document': doc,
                        'metadata': meta,
                        'similarity_percent': (1 - dist) * 100,
                        'chunk_source': meta.get('source', 'Unknown'),
                        'reranked': False
                    })
            
            return formatted_results[:n_results]
            
        except Exception as e:
            logger.error(f"❌ Search error: {e}")
            return []

    def multi_query_retrieval(self, query: str, n_results_per_query: int = 3, max_total_results: int = 5) -> List[Dict[str, Any]]:
        query_variations = self.expand_query(query)
        all_results = []
        seen_docs = set()
        
        for q_var in query_variations:
            results = self.similarity_search(q_var, n_results=n_results_per_query)
            for res in results:
                doc_hash = hash(res['document'][:150])
                if doc_hash not in seen_docs:
                    seen_docs.add(doc_hash)
                    all_results.append(res)
        
        if self.cross_encoder and len(all_results) > 1:
            pairs = [(query, res['document']) for res in all_results]
            scores = self.cross_encoder.predict(pairs)
            
            for res, score in zip(all_results, scores):
                res['final_score'] = float(score)
                res['similarity_percent'] = float(score) * 100
            
            all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        min_confidence_percent = self.min_confidence * 100
        filtered_results = [
            r for r in all_results 
            if r.get('similarity_percent', 0) > min_confidence_percent
        ]
        
        if not filtered_results:
            logger.warning(f"⚠️ No results above confidence threshold {min_confidence_percent}%")
            return all_results[:max_total_results]
        
        logger.info(f"✅ Multi-query retrieval: {len(filtered_results)} hasil (dari {len(all_results)} total)")
        return filtered_results[:max_total_results]

    def hybrid_search(self, query: str, n_results: int = 5, alpha: float = 0.6) -> List[Dict[str, Any]]:
        if not self.use_hybrid_search or not self.bm25:
            return self.similarity_search(query, n_results=n_results)
        
        try:
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results * 3, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            if not semantic_results['documents'][0]:
                return []
            
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            semantic_scores = [1 - d for d in semantic_results['distances'][0]]
            max_sem = max(semantic_scores) if semantic_scores else 1
            semantic_scores_norm = np.array(semantic_scores) / max_sem
            
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores_norm = bm25_scores / max_bm25
            
            combined_results = []
            for i, (doc, meta, sem_score) in enumerate(zip(
                semantic_results['documents'][0],
                semantic_results['metadatas'][0],
                semantic_scores_norm
            )):
                try:
                    doc_idx = self.corpus.index(doc)
                    bm25_score = bm25_scores_norm[doc_idx]
                except (ValueError, IndexError):
                    bm25_score = 0
                
                final_score = alpha * sem_score + (1 - alpha) * bm25_score
                
                combined_results.append({
                    'document': doc,
                    'metadata': meta,
                    'final_score': float(final_score),
                    'similarity_percent': float(final_score) * 100,
                    'semantic_score': float(sem_score),
                    'bm25_score': float(bm25_score),
                    'chunk_source': meta.get('source', 'Unknown'),
                    'method': 'hybrid'
                })
            
            combined_results.sort(key=lambda x: x['final_score'], reverse=True)
            min_confidence_percent = self.min_confidence * 100
            filtered = [r for r in combined_results if r['similarity_percent'] > min_confidence_percent]
            
            return filtered[:n_results]
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            return self.similarity_search(query, n_results=n_results)

    def adaptive_retrieval(self, query: str) -> List[Dict[str, Any]]:
        query_length = len(query.split())
        if query_length < 5:
            return self.hybrid_search(query, n_results=3, alpha=0.5)
        elif query_length <= 10:
            return self.multi_query_retrieval(query, max_total_results=4)
        else:
            return self.multi_query_retrieval(query, max_total_results=5)

    # ========================================================================
    # RESPONSE GENERATION
    # ========================================================================

    def compress_context(self, results: List[Dict[str, Any]], max_tokens: int = 2000) -> str:
        context = ""
        token_count = 0
        for i, res in enumerate(results, 1):
            chunk = f"[SUMBER {i}: {res['chunk_source']}]\n{res['document']}\n\n"
            chunk_tokens = len(chunk.split())
            if token_count + chunk_tokens < max_tokens:
                context += chunk
                token_count += chunk_tokens
            else:
                break
        return context

    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        context_text = self.compress_context(search_results, max_tokens=2000)
        
        if not context_text:
            context_text = "STATUS DATABASE: KOSONG. Tidak ada dokumen yang relevan ditarik."

        system_prompt = f"""Anda adalah AgriBot, Asisten Ahli Pertanian Indonesia yang cerdas, praktis, dan profesional. Fokus Anda: Kentang, Cabai, dan Jagung.

KONTEKS DATABASE (Ditarik berdasarkan kemiripan vektor):
{context_text}

ATURAN BERPIKIR & MENJAWAB:
1. Pahami Niat Pengguna (Intent):
   - JIKA pengguna bertanya TEORI/INFORMASI UMUM (misal: "Apa saja penyakit cabai?", "Sebutkan hama kentang"): Jawablah secara langsung, informatif, dan terstruktur. JANGAN berasumsi tanaman mereka sakit. JANGAN berikan ucapan simpati.
   - JIKA pengguna MENGELUHKAN masalah (misal: "Daun cabai saya keriting", "Tolong tanaman saya layu"): Berikan sedikit simpati natural di awal kalimat, lalu berikan solusi.

2. Evaluasi Konteks: 
   - Gunakan KONTEKS DATABASE sebagai sumber utama. 
   - Jika konteks membahas hal yang BERBEDA dengan pertanyaan (misal: ditanya cabai, tapi konteks berisi kentang), ABAIKAN konteks tersebut.
   - Jika STATUS DATABASE: KOSONG atau diabaikan, jawab menggunakan pengetahuan umum Anda secara ringkas dan akurat, TAPI tambahkan catatan kecil bahwa "berdasarkan pengetahuan umum" (jangan berhalusinasi mengutip database).

3. Gaya Bahasa: Mengalir, natural, dan to-the-point. Gunakan bullet points untuk kemudahan membaca. Hindari basa-basi yang terlalu panjang.

4. Call to Action (Khusus Keluhan): HANYA JIKA pengguna mengeluhkan tanaman sakit, sarankan mereka untuk mengunggah foto di Tab "Identifikasi Foto" agar sistem Vision dapat menganalisisnya. Jika pertanyaan umum, abaikan saran ini.

PERTANYAAN PENGGUNA: {query}"""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                model=self.llm_model_name,
                temperature=0.3,
                max_tokens=1024,
            )
            
            response = chat_completion.choices[0].message.content
            logger.info(f"✅ Response generated ({len(response.split())} words)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return f"Maaf, saya sedang mengalami kendala teknis saat memproses jawaban: {str(e)}"

    def _generate_social_response(self, query: str) -> str:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": """Anda adalah AgriBot, asisten pertanian yang ramah. 
                        Tugas Anda:
                        1. Jika ditanya kabar: Jawab bahwa Anda baik baik saja dan siap membantu petani.
                        2. Jika dipuji: Berterima kasih dengan rendah hati.
                        3. Jika disapa: Balas dengan sapaan yang hangat sesuai waktu (pagi/siang/malam).
                        4. Selalu akhiri dengan menawarkan bantuan spesifik tentang tanaman kentang, cabai, atau jagung.
                        Jawab maksimal 2-3 kalimat agar tetap ringkas."""
                    },
                    {"role": "user", "content": query}
                ],
                model=self.llm_model_name,
                temperature=0.8, # Naikkan dikit agar tidak kaku
                max_tokens=150,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Social response error: {e}")
            return "Halo! Kabar saya baik dan siap membantu Anda. Ada yang bisa saya bantu terkait tanaman kentang, cabai, atau jagung hari ini?"

    def _is_social_query(self, query: str) -> bool:
        social_keywords = [
            "halo", "hai", "hello", "hi", "selamat", "pagi", "siang", 
            "sore", "malam", "thanks", "makasih", "terima kasih"
        ]
        query_lower = query.lower()
        has_keyword = any(k in query_lower for k in social_keywords)
        is_short = len(query.split()) < 5
        return has_keyword and is_short
    
    def _get_intent(self, query: str) -> str:
        query_lower = query.lower()
        identify_keywords = ["coba", "identifikasi", "periksa", "cek", "deteksi", "analisis"]
        if any(k in query_lower for k in identify_keywords) and len(query_lower.split()) < 6:
            return "INTENT_COBA"
            
        # 2. Intent Teknis Tanaman (Pemicu RAG)
        # Fokus pada komoditas utama proyek LEAP Anda: Kentang, Cabai, Jagung
        agri_keywords = [
            "kentang", "cabai", "cabe", "jagung", "daun", "batang", "akar", "bulai"
            "hama", "penyakit", "bercak", "layu", "busuk", "ulat", "fungisida", 
            "pestisida", "pupuk", "tanam", "panen", "phytophthora", "fusarium"
        ]
        if any(k in query_lower for k in agri_keywords):
            return "INTENT_RAG_TECHNICAL"
            
        # 3. Default: Masuk ke General LLM (Social & Pengetahuan Umum lainnya)
        return "INTENT_GENERAL_LLM"
    
    def _generate_general_response(self, query: str) -> str:
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": """Anda adalah AgriBot. 
                        - Untuk sapaan/kabar: Jawab dengan ramah, ceria, dan singkat.
                        - Untuk pertanyaan umum non-pertanian: Jawablah dengan cerdas namun tetap ingatkan bahwa spesialisasi utama Anda adalah membantu petani kentang, cabai, dan jagung.
                        - Bahasa: Indonesia yang natural (tidak kaku)."""
                    },
                    {"role": "user", "content": query}
                ],
                model=self.llm_model_name,
                temperature=0.7,
                max_tokens=512,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ General response error: {e}")
            return "Halo! Ada yang bisa saya bantu mengenai tanaman Anda hari ini?"

    def transcribe_audio(self, audio_bytes: bytes) -> str:
        """Mengonversi audio ke teks menggunakan Groq Whisper"""
        try:
            # Gunakan buffer io untuk mengirim file ke API
            audio_file = ("speech.wav", audio_bytes)
            transcription = self.groq_client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-large-v3",
                language="id", # Mengunci ke Bahasa Indonesia
                response_format="text"
            )
            return transcription
        except Exception as e:
            logger.error(f"❌ Transkripsi gagal: {e}")
            return ""

    def chat(self, query: str, retrieval_mode: str = "adaptive") -> Dict[str, Any]:
        logger.info(f"🗣️ User Query: {query}")
        intent = self._get_intent(query)
        
        # JALUR A: ACTION UI
        if intent == "INTENT_COBA":
            return {
                'response': "Tentu! Silakan unggah foto daun di tab 'Identifikasi Foto' untuk saya cek.",
                'search_results': [],
                'metadata': {'type': 'ui_action'}
            }

        # JALUR B: RAG (Hanya untuk Teknis Tanaman)
        if intent == "INTENT_RAG_TECHNICAL":
            corrected_query = self.correct_typos(query)
            search_results = self.adaptive_retrieval(corrected_query)
            response = self.generate_response(corrected_query, search_results)
            
            return {
                'response': response,
                'search_results': search_results,
                'metadata': {'type': 'rag_technical', 'intent': intent}
            }

        # JALUR C: GENERAL LLM (Social, Kabar, Pertanyaan Umum non-tanaman)
        # Langsung panggil LLM tanpa context database
        response = self._generate_general_response(query)
        return {
            'response': response,
            'search_results': [],
            'metadata': {'type': 'general_llm', 'intent': intent}
        }
    
    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleared")