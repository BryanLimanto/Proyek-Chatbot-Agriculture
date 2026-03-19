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
    """
    Advanced RAG Chatbot dengan Multiple Retrieval Strategies
    
    Features:
    - Multi-Query Retrieval (Query Expansion)
    - Hybrid Search (Semantic + BM25)
    - Cross-Encoder Reranking
    - Confidence Threshold Filtering
    - Adaptive Context Management
    """
    
    def __init__(
        self, 
        chroma_dir: str, 
        collection_name: str, 
        groq_api_key: str, 
        cross_encoder_model: str = "BAAI/bge-reranker-base",
        min_confidence: float = 0.30,
        use_hybrid_search: bool = True
    ):
        """
        Initialize chatbot dengan konfigurasi advanced
        
        Args:
            chroma_dir: Path ke ChromaDB directory
            collection_name: Nama collection di ChromaDB
            groq_api_key: API key untuk Groq
            cross_encoder_model: Model untuk reranking
            min_confidence: Minimum confidence threshold (0-1)
            use_hybrid_search: Enable hybrid search (semantic + BM25)
        """
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.cross_encoder_model = cross_encoder_model
        self.min_confidence = min_confidence
        self.use_hybrid_search = use_hybrid_search
        
        # Groq Client
        self.groq_client = Groq(api_key=groq_api_key)
        self.llm_model_name = "llama-3.3-70b-versatile"  # Fast & good balance
        
        # Embedding Function (MUST match ingestion!)
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
        """Initialize ChromaDB client dan collection"""
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
        """Load Cross-Encoder untuk reranking"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(self.cross_encoder_model, device=device)
            logger.info(f"✅ Cross-encoder dimuat di {device}")
        except Exception as e:
            logger.warning(f"⚠️ Gagal memuat Cross-Encoder: {e}. Lanjut tanpa reranking.")
            self.cross_encoder = None

    def _init_symspell(self):
        """Initialize SymSpell untuk typo correction"""
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self._create_default_dictionary()

    def _create_default_dictionary(self):
        """Kamus sederhana untuk fallback"""
        words = [
            "kentang", "jagung", "cabai", "tomat", "padi", "gandum",
            "penyakit", "daun", "batang", "akar", "buah", "bunga",
            "obat", "pupuk", "pestisida", "fungisida", "herbisida",
            "tanaman", "pertanian", "petani", "hama", "virus", "bakteri", "jamur"
        ]
        for word in words:
            self.sym_spell.create_dictionary_entry(word, 1)

    def _init_bm25(self):
        """Initialize BM25 untuk hybrid search"""
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
        """Simple typo correction"""
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
        """
        Generate multiple query variations using LLM
        
        Returns:
            List of query variations (including original)
        """
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": """Anda adalah ahli pertanian. Buatlah 3 variasi pertanyaan untuk mendapatkan informasi yang lebih lengkap dari database.

ATURAN:
1. Fokus pada aspek berbeda: penyebab, gejala, solusi, pencegahan
2. Gunakan sinonim dan istilah teknis pertanian
3. Format: satu baris per variasi, tanpa numbering atau bullet
4. Tetap dalam konteks pertanian/penyakit tanaman

Contoh Input: "Daun kentang saya menguning"
Output:
Apa penyebab daun kentang berubah warna menjadi kuning?
Bagaimana cara mengobati klorosis pada tanaman kentang?
Penyakit atau defisiensi nutrisi apa yang membuat daun kentang menguning?"""
                    },
                    {"role": "user", "content": f"Query: {original_query}"}
                ],
                model=self.llm_model_name,
                temperature=0.7,
                max_tokens=200,
            )
            
            variations = chat_completion.choices[0].message.content.strip().split('\n')
            variations = [v.strip() for v in variations if v.strip() and len(v.strip()) > 10]
            
            # Gabung dengan query asli
            all_queries = [original_query] + variations[:3]
            logger.info(f"🔍 Query expansion: {len(all_queries)} variations")
            return all_queries
            
        except Exception as e:
            logger.error(f"❌ Query expansion gagal: {e}")
            return [original_query]

    def similarity_search(
        self, 
        query: str, 
        n_results: int = 5, 
        initial_candidates: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Basic semantic similarity search dengan reranking
        
        Args:
            query: Search query
            n_results: Jumlah hasil akhir
            initial_candidates: Jumlah candidates sebelum reranking
            
        Returns:
            List of search results dengan metadata
        """
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
            
            # Reranking dengan Cross-Encoder
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
                # Fallback: cosine similarity
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

    def multi_query_retrieval(
        self, 
        query: str, 
        n_results_per_query: int = 3,
        max_total_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        🌟 METODE UTAMA: Multi-Query Retrieval dengan deduplication
        
        Args:
            query: Original user query
            n_results_per_query: Results per query variation
            max_total_results: Maximum final results
            
        Returns:
            Deduplicated and reranked results
        """
        # 1. Expand query
        query_variations = self.expand_query(query)
        
        # 2. Retrieve untuk setiap variasi
        all_results = []
        seen_docs = set()
        
        for q_var in query_variations:
            results = self.similarity_search(q_var, n_results=n_results_per_query)
            
            for res in results:
                # Deduplication based on first 150 chars
                doc_hash = hash(res['document'][:150])
                if doc_hash not in seen_docs:
                    seen_docs.add(doc_hash)
                    all_results.append(res)
        
        # 3. Re-rank semua hasil dengan original query
        if self.cross_encoder and len(all_results) > 1:
            pairs = [(query, res['document']) for res in all_results]
            scores = self.cross_encoder.predict(pairs)
            
            for res, score in zip(all_results, scores):
                res['final_score'] = float(score)
                res['similarity_percent'] = float(score) * 100
            
            all_results.sort(key=lambda x: x['final_score'], reverse=True)
        
        # 4. Filter by confidence threshold
        min_confidence_percent = self.min_confidence * 100
        filtered_results = [
            r for r in all_results 
            if r.get('similarity_percent', 0) > min_confidence_percent
        ]
        
        if not filtered_results:
            logger.warning(f"⚠️ No results above confidence threshold {min_confidence_percent}%")
            return all_results[:max_total_results]  # Return top results anyway
        
        logger.info(f"✅ Multi-query retrieval: {len(filtered_results)} hasil (dari {len(all_results)} total)")
        return filtered_results[:max_total_results]

    def hybrid_search(
        self, 
        query: str, 
        n_results: int = 5, 
        alpha: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search: Semantic (ChromaDB) + Keyword (BM25)
        
        Args:
            query: Search query
            n_results: Number of results
            alpha: Weight for semantic search (0-1). 
                   Higher = more semantic, Lower = more keyword
                   
        Returns:
            Combined and ranked results
        """
        if not self.use_hybrid_search or not self.bm25:
            logger.warning("⚠️ BM25 tidak tersedia, fallback ke semantic search")
            return self.similarity_search(query, n_results=n_results)
        
        try:
            # 1. Semantic Search
            semantic_results = self.collection.query(
                query_texts=[query],
                n_results=min(n_results * 3, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            if not semantic_results['documents'][0]:
                return []
            
            # 2. BM25 Search
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # 3. Normalize scores
            semantic_scores = [1 - d for d in semantic_results['distances'][0]]
            max_sem = max(semantic_scores) if semantic_scores else 1
            semantic_scores_norm = np.array(semantic_scores) / max_sem
            
            max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
            bm25_scores_norm = bm25_scores / max_bm25
            
            # 4. Combine scores
            combined_results = []
            for i, (doc, meta, sem_score) in enumerate(zip(
                semantic_results['documents'][0],
                semantic_results['metadatas'][0],
                semantic_scores_norm
            )):
                # Find BM25 score untuk doc yang sama
                try:
                    doc_idx = self.corpus.index(doc)
                    bm25_score = bm25_scores_norm[doc_idx]
                except (ValueError, IndexError):
                    bm25_score = 0
                
                # Weighted combination
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
            
            # 5. Sort by final score
            combined_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 6. Filter by confidence
            min_confidence_percent = self.min_confidence * 100
            filtered = [r for r in combined_results if r['similarity_percent'] > min_confidence_percent]
            
            logger.info(f"✅ Hybrid search: {len(filtered)} hasil (alpha={alpha})")
            return filtered[:n_results]
            
        except Exception as e:
            logger.error(f"❌ Hybrid search error: {e}")
            return self.similarity_search(query, n_results=n_results)

    def adaptive_retrieval(self, query: str) -> List[Dict[str, Any]]:
        """
        🎯 METODE OTOMATIS: Pilih strategi retrieval based on query
        
        Returns:
            Best results using appropriate strategy
        """
        query_length = len(query.split())
        
        # Query pendek (< 5 kata) → Hybrid search (keyword penting)
        if query_length < 5:
            logger.info("📍 Strategi: Hybrid search (query pendek)")
            return self.hybrid_search(query, n_results=3, alpha=0.5)
        
        # Query medium (5-10 kata) → Multi-query retrieval
        elif query_length <= 10:
            logger.info("📍 Strategi: Multi-query retrieval (query medium)")
            return self.multi_query_retrieval(query, max_total_results=4)
        
        # Query panjang (> 10 kata) → Multi-query dengan lebih banyak hasil
        else:
            logger.info("📍 Strategi: Multi-query retrieval (query panjang)")
            return self.multi_query_retrieval(query, max_total_results=5)

    # ========================================================================
    # RESPONSE GENERATION
    # ========================================================================

    def compress_context(
        self, 
        results: List[Dict[str, Any]], 
        max_tokens: int = 2000
    ) -> str:
        """
        Compress context to fit token limit
        
        Args:
            results: Search results
            max_tokens: Maximum tokens (approximate by words)
            
        Returns:
            Compressed context string
        """
        context = ""
        token_count = 0
        
        for i, res in enumerate(results, 1):
            chunk = f"[SUMBER {i}: {res['chunk_source']}]\n{res['document']}\n\n"
            chunk_tokens = len(chunk.split())
            
            if token_count + chunk_tokens < max_tokens:
                context += chunk
                token_count += chunk_tokens
            else:
                logger.info(f"⚠️ Context dipotong di chunk {i} (token limit)")
                break
        
        return context

    def generate_response(
        self, 
        query: str, 
        search_results: List[Dict[str, Any]]
    ) -> str:
        """
        Generate response using Groq LLM with retrieved context
        
        Args:
            query: User query
            search_results: Retrieved documents
            
        Returns:
            Generated response
        """
        # 1. Compress context
        context_text = self.compress_context(search_results, max_tokens=2000)
        
        if not context_text:
            context_text = "Tidak ada informasi spesifik yang ditemukan di database."

        # 2. Build system prompt
        system_prompt = f"""Anda adalah Asisten Ahli Pertanian Indonesia yang ramah dan profesional.

KONTEKS DARI DATABASE:
{context_text}

ATURAN PENTING:
1. Jawab HANYA berdasarkan konteks di atas. Jika tidak ada info, katakan dengan jelas.
2. Gunakan Bahasa Indonesia yang mudah dipahami petani.
3. Struktur jawaban:
   - Identifikasi masalah (jika ada)
   - Penyebab/penjelasan
   - Solusi konkret (langkah-langkah atau rekomendasi)
4. Gunakan bullet points untuk daftar/langkah-langkah
5. Sebutkan sumber jika relevan (misal: "Menurut panduan XYZ...")
6. JANGAN membuat informasi yang tidak ada di konteks
7. Jika konteks tidak cukup, sarankan user untuk lebih spesifik

GAYA BAHASA:
- Ramah tapi profesional
- Gunakan istilah teknis + penjelasan sederhana
- Contoh: "Early blight (hawar daun awal) adalah..."
"""

        try:
            # 3. Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                model=self.llm_model_name,
                temperature=0.3,  # Lower = more focused on context
                max_tokens=1024,
            )
            
            response = chat_completion.choices[0].message.content
            logger.info(f"✅ Response generated ({len(response.split())} words)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return f"Maaf, terjadi kesalahan saat memproses jawaban: {str(e)}"

    def _generate_social_response(self, query: str) -> str:
        """Generate response untuk social queries (greetings, dll)"""
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system", 
                        "content": "Anda adalah asisten pertanian yang ramah. Jawab sapaan dengan singkat dan natural (1-2 kalimat)."
                    },
                    {"role": "user", "content": query}
                ],
                model=self.llm_model_name,
                temperature=0.8,
                max_tokens=100,
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ Social response error: {e}")
            return "Halo! Ada yang bisa saya bantu tentang tanaman Anda?"

    def _is_social_query(self, query: str) -> bool:
        """Check if query is social/greeting"""
        social_keywords = [
            "halo", "hai", "hello", "hi", "selamat", "pagi", "siang", 
            "sore", "malam", "thanks", "makasih", "terima kasih"
        ]
        query_lower = query.lower()
        
        # Social jika: mengandung keyword DAN pendek
        has_keyword = any(k in query_lower for k in social_keywords)
        is_short = len(query.split()) < 5
        
        return has_keyword and is_short
    
    def _get_intent(self, query: str) -> str:
        """
        Deteksi Intent menggunakan pola Token sederhana (Automata State)
        """
        query_lower = query.lower()
        
        # 1. Intent: Coba / Identifikasi (Trigger untuk Vision nanti)
        identify_keywords = ["coba", "identifikasi", "periksa", "cek", "deteksi", "analisis"]
        if any(k in query_lower for k in identify_keywords) and len(query_lower.split()) < 6:
            return "INTENT_COBA"
            
        # 2. Intent: Social / Greeting
        social_keywords = ["halo", "hai", "siapa kamu", "pagi", "siang", "malam", "assalamualaikum"]
        if any(k in query_lower for k in social_keywords) and len(query_lower.split()) < 4:
            return "INTENT_SOCIAL"
            
        # 3. Intent: Knowledge (Default RAG)
        return "INTENT_KNOWLEDGE"

    def chat(self, query: str, retrieval_mode: str = "adaptive") -> Dict[str, Any]:
        logger.info(f"🗣️ User Query: {query}")
        
        # 1. Koreksi Typo
        query = self.correct_typos(query)
        
        # 2. Deteksi Intent (Automata State)
        intent = self._get_intent(query)
        
        if intent == "INTENT_SOCIAL":
            return {
                'response': self._generate_social_response(query),
                'search_results': [],
                'metadata': {'query_type': 'social', 'intent': intent}
            }

        if intent == "INTENT_COBA":
            return {
                'response': "Tentu! Silakan unggah foto daun di tab 'Analisis Foto Daun'.",
                'search_results': [],
                'metadata': {'query_type': 'intent_action', 'intent': intent}
            }

        # 3. Alur Knowledge (RAG)
        # Gunakan adaptive_retrieval agar sistem pintar memilih Hybrid/Semantic
        search_results = self.adaptive_retrieval(query)
        
        # Jika hasil RAG kosong, bot akan menjawab gagal
        if not search_results:
            logger.warning("🔍 RAG gagal menemukan dokumen yang relevan.")
            return {
                'response': "Maaf, saya tidak menemukan informasi tentang itu di dataset Kentang, Cabai, dan Jagung saya.",
                'search_results': [],
                'metadata': {'query_type': 'no_results', 'intent': intent}
            }

        # 4. Generate Jawaban dari LLM (Groq)
        response = self.generate_response(query, search_results)
        
        return {
            'response': response,
            'search_results': search_results,
            'metadata': {
                'query_type': 'knowledge', 
                'intent': intent,
                'top_score': search_results[0]['similarity_percent']
            }
        }

    # ========================================================================
    # MAIN CHAT INTERFACE
    # ========================================================================

    def chat(
        self, 
        query: str, 
        retrieval_mode: str = "adaptive",
        n_results: int = 4,
        use_query_expansion: bool = True
    ) -> Dict[str, Any]:
        """
        🎯 MAIN INTERFACE: Chat dengan user
        
        Args:
            query: User query
            retrieval_mode: "adaptive", "multi_query", "hybrid", or "basic"
            n_results: Jumlah chunks untuk context
            use_query_expansion: Enable query expansion
            
        Returns:
            Dict dengan 'response', 'search_results', dan 'metadata'
        """
        logger.info(f"🗣️ Query: {query}")
        logger.info(f"⚙️ Mode: {retrieval_mode}, n_results: {n_results}")
        
        # 1. Check social query
        if self._is_social_query(query):
            return {
                'response': self._generate_social_response(query),
                'search_results': [],
                'metadata': {'query_type': 'social', 'retrieval_mode': 'none'}
            }

        # 2. Retrieval based on mode
        if retrieval_mode == "adaptive":
            search_results = self.adaptive_retrieval(query)
        elif retrieval_mode == "multi_query" and use_query_expansion:
            search_results = self.multi_query_retrieval(query, max_total_results=n_results)
        elif retrieval_mode == "hybrid":
            search_results = self.hybrid_search(query, n_results=n_results)
        else:  # basic
            search_results = self.similarity_search(query, n_results=n_results)
        
        # 3. Check if no results
        if not search_results:
            return {
                'response': "Maaf, saya tidak menemukan informasi yang relevan di database. Bisa tolong perjelas pertanyaan Anda?",
                'search_results': [],
                'metadata': {'query_type': 'no_results', 'retrieval_mode': retrieval_mode}
            }
        
        # 4. Generate response
        response = self.generate_response(query, search_results)
        
        # 5. Metadata untuk debugging/analytics
        metadata = {
            'query_type': 'knowledge',
            'retrieval_mode': retrieval_mode,
            'num_results': len(search_results),
            'top_confidence': search_results[0].get('similarity_percent', 0) if search_results else 0,
            'avg_confidence': np.mean([r.get('similarity_percent', 0) for r in search_results]) if search_results else 0
        }
        
        logger.info(f"✅ Chat selesai - Top confidence: {metadata['top_confidence']:.1f}%")
        
        return {
            'response': response,
            'search_results': search_results,
            'metadata': metadata
        }

    def batch_chat(self, queries: List[str], retrieval_mode: str = "adaptive") -> List[Dict[str, Any]]:
        """Process multiple queries sekaligus"""
        results = []
        for query in queries:
            result = self.chat(query, retrieval_mode=retrieval_mode)
            results.append(result)
        return results

    def clear_memory(self):
        """Clear GPU memory jika ada"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("🧹 GPU memory cleared")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def test_chatbot(chatbot: KnowledgeBaseChatbot, test_queries: List[str] = None):
    """
    Test chatbot dengan berbagai query
    
    Usage:
        test_chatbot(chatbot)
    """
    if test_queries is None:
        test_queries = [
            "Daun kentang saya menguning dan ada bercak coklat",
            "Bagaimana cara mengobati hawar daun pada kentang?",
            "Pupuk apa yang bagus untuk kentang?",
            "Halo selamat pagi"
        ]
    
    print("="*80)
    print("🧪 TESTING CHATBOT")
    print("="*80)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}: {query}")
        print('='*80)
        
        # Test dengan mode adaptive
        result = chatbot.chat(query, retrieval_mode="adaptive")
        
        print(f"\n📊 METADATA:")
        print(f"   - Mode: {result['metadata']['retrieval_mode']}")
        print(f"   - Results: {result['metadata']['num_results']}")
        print(f"   - Top Confidence: {result['metadata']['top_confidence']:.1f}%")
        
        if result['search_results']:
            print(f"\n📄 TOP SOURCES:")
            for j, res in enumerate(result['search_results'][:3], 1):
                print(f"   {j}. {res['chunk_source']} ({res['similarity_percent']:.1f}%)")
        
        print(f"\n💬 RESPONSE:")
        print(result['response'])
        print()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example: Cara menggunakan chatbot
    """
    
    # Initialize chatbot
    chatbot = KnowledgeBaseChatbot(
        chroma_dir="./chroma_db",
        collection_name="agricultural_knowledge",
        groq_api_key="your-groq-api-key-here",
        min_confidence=0.30,  # 30% minimum confidence
        use_hybrid_search=True  # Enable hybrid search
    )
    
    # Single query
    query = "Daun kentang saya menguning, apa penyebabnya?"
    result = chatbot.chat(query, retrieval_mode="adaptive")
    print(result['response'])
    
    # Batch queries
    queries = [
        "Bagaimana cara mencegah penyakit hawar daun?",
        "Pupuk organik apa yang cocok untuk kentang?",
        "Kapan waktu terbaik menanam jagung?"
    ]
    results = chatbot.batch_chat(queries)
    
    # Test chatbot
    test_chatbot(chatbot)