import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import logging
import re
from symspellpy import SymSpell, Verbosity
import numpy as np
from sentence_transformers import CrossEncoder

# ============================================================================
# KONFIGURASI DAN INISIALISASI
# ============================================================================

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseChatbot:
    def __init__(self, chroma_dir: str, collection_name: str, model_path: str, 
                 cross_encoder_model: str = "cross-encoder/mmarco-mMiniLM-v6-translated-gpu"):
        
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.model_path = model_path
        self.cross_encoder_model = cross_encoder_model

        # 1. DEFINISIKAN EMBEDDING FUNCTION YANG SAMA DENGAN SAAT INGEST
        # Ini wajib ada agar Query user diubah jadi angka dengan rumus yang sama
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # Inisialisasi komponen
        self._init_chroma_client()
        self._load_llm_model()
        self._init_symspell()
        self._load_cross_encoder()
        
        self.key_terms = set()
        logger.info("Chatbot berhasil diinisialisasi")

    # Perbaiki fungsi ini untuk menggunakan embedding_function
    def _init_chroma_client(self):
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # WAJIB MEMASUKKAN embedding_function DI SINI
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn,  # <--- INI KUNCINYA
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB collection '{self.collection_name}' berhasil diakses")
        except Exception as e:
            logger.error(f"Error inisialisasi ChromaDB: {e}")
            raise
    
    # ============================================================================
    # FUNGSI INISIALISASI CROSS-ENCODER
    # ============================================================================
    
    def _load_cross_encoder(self):
        try:
            logger.info(f"Memuat cross-encoder: {self.cross_encoder_model}")
            # Coba load ke CPU dulu jika GPU bermasalah, atau force device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(self.cross_encoder_model, device=device)
            logger.info(f"Cross-encoder berhasil dimuat di {device}")
        except Exception as e:
            logger.error(f"GAGAL memuat Cross-Encoder: {e}")
            logger.warning("⚠️ Fitur Reranking dinonaktifkan. Menggunakan Cosine Similarity standar.")
            self.cross_encoder = None
    
    # ============================================================================
    # FUNGSI INISIALISASI SYMSPELL
    # ============================================================================
    
    def _init_symspell(self):
        """Inisialisasi SymSpell untuk typo correction"""
        try:
            # Buat instance SymSpell
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            
            # Coba load kamus dari file (jika ada)
            dictionary_path = "./dataset/indonesian_dictionary.txt"
            if os.path.exists(dictionary_path):
                # Load kamus khusus Bahasa Indonesia jika ada
                self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
                logger.info(f"Kamus Bahasa Indonesia dimuat dari {dictionary_path}")
            else:
                # Jika tidak ada kamus khusus, buat kamus dari teks default
                self._create_default_dictionary()
                logger.info("Kamus default dibuat")
                
        except Exception as e:
            logger.error(f"Error inisialisasi SymSpell: {e}")
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    def _create_default_dictionary(self):
        """Buat kamus default dengan kata-kata umum Bahasa Indonesia dan Inggris"""
        indonesian_words = [
            "apa", "bagaimana", "dimana", "kapan", "siapa", "mengapa",
            "dokumen", "informasi", "data", "sistem", "program", "aplikasi",
            "perusahaan", "karyawan", "manajemen", "proyek", "laporan",
            "teknologi", "komputer", "jaringan", "internet", "database",
            "indonesia", "jakarta", "surabaya", "bandung", "medan"
        ]
        
        english_words = [
            "chatbot", "ai", "machine", "learning", "deep", "neural", "network",
            "algorithm", "python", "javascript", "java", "html", "css",
            "database", "server", "client", "api", "rest", "graphql",
            "cloud", "aws", "azure", "google", "microsoft", "apple"
        ]
        
        all_words = indonesian_words + english_words
        
        for word in all_words:
            self.sym_spell.create_dictionary_entry(word, 1)
    
    # ============================================================================
    # FUNGSI TYPO CORRECTION
    # ============================================================================
    
    def correct_typos(self, text: str, max_edit_distance: int = 2) -> str:
        """Koreksi typo dalam teks menggunakan SymSpell"""
        try:
            words = text.split()
            corrected_words = []
            
            for word in words:
                suggestions = self.sym_spell.lookup(
                    word, 
                    Verbosity.CLOSEST, 
                    max_edit_distance=max_edit_distance,
                    include_unknown=True
                )
                
                if suggestions:
                    best_suggestion = suggestions[0]
                    
                    if best_suggestion.distance == 0:
                        corrected_words.append(word)
                    elif best_suggestion.distance <= max_edit_distance and best_suggestion.count > 0:
                        corrected_words.append(best_suggestion.term)
                        logger.debug(f"Koreksi: '{word}' -> '{best_suggestion.term}'")
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
            
            corrected_text = " ".join(corrected_words)
            
            if corrected_text != text:
                logger.info(f"Koreksi typo: '{text}' -> '{corrected_text}'")
            
            return corrected_text
            
        except Exception as e:
            logger.error(f"Error dalam koreksi typo: {e}")
            return text
    
    # ============================================================================
    # FUNGSI INISIALISASI CHROMADB
    # ============================================================================
    
    def _init_chroma_client(self):
        """Inisialisasi ChromaDB client dan koleksi"""
        try:
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB collection '{self.collection_name}' berhasil diakses")
            
        except Exception as e:
            logger.error(f"Error inisialisasi ChromaDB: {e}")
            raise
    
    # ============================================================================
    # FUNGSI LOAD MODEL LLM
    # ============================================================================
    
    def _load_llm_model(self):
        """Load model LLM dengan Kuantisasi 4-bit (Agar muat di GPU 4GB)"""
        try:
            logger.info(f"Memuat model LLM dari: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # === BAGIAN PENTING: KONFIGURASI 4-BIT ===
            # Ini yang akan mengecilkan ukuran model dari 6GB menjadi ~2.5GB
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            # =========================================

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,  # <--- Menggunakan config di atas
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True           # <--- Mencegah RAM 8GB penuh saat loading
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model LLM berhasil dimuat dalam mode 4-bit (Hemat VRAM)")
            
        except Exception as e:
            logger.error(f"Error memuat model LLM: {e}")
            raise
    
    # ============================================================================
    # FUNGSI SIMILARITY SEARCH DENGAN CROSS-ENCODER (DIPERBAIKI)
    # ============================================================================
    
    def similarity_search(self, query: str, n_results: int = 5, 
                         initial_candidates: int = 20, rerank_top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Melakukan similarity search dengan cross-encoder reranking
        
        Args:
            query: Query dari user
            n_results: Jumlah hasil akhir yang ingin diambil
            initial_candidates: Jumlah kandidat awal dari ChromaDB
            rerank_top_k: Jumlah kandidat untuk di-rerank dengan cross-encoder
            
        Returns:
            List hasil similarity search yang sudah di-rerank
        """
        try:
            # Step 1: Ambil lebih banyak kandidat awal dari ChromaDB
            logger.info(f"Mengambil {initial_candidates} kandidat awal dari ChromaDB")
            initial_results = self.collection.query(
                query_texts=[query],
                n_results=min(initial_candidates, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            if not initial_results['documents'] or not initial_results['documents'][0]:
                logger.warning("Tidak ada dokumen ditemukan di ChromaDB")
                return []
            
            documents = initial_results['documents'][0]
            metadatas = initial_results['metadatas'][0]
            distances = initial_results['distances'][0]
            
            # Step 2: Rerank dengan cross-encoder jika tersedia
            if self.cross_encoder is not None and len(documents) > 1:
                logger.info(f"Reranking {min(rerank_top_k, len(documents))} kandidat dengan cross-encoder")
                
                # Siapkan pasangan query-dokumen untuk cross-encoder
                pairs = [(query, doc) for doc in documents[:rerank_top_k]]
                
                # Dapatkan skor dari cross-encoder
                cross_encoder_scores = self.cross_encoder.predict(pairs)
                
                # Gabungkan skor cross-encoder dengan cosine similarity
                combined_scores = []
                for i, (doc, metadata, cos_distance, ce_score) in enumerate(
                    zip(documents[:rerank_top_k], metadatas[:rerank_top_k], 
                        distances[:rerank_top_k], cross_encoder_scores)
                ):
                    # Normalisasi skor cross-encoder (biasanya sudah antara 0-1)
                    ce_score_normalized = float(ce_score)
                    
                    # Konversi cosine distance ke similarity
                    cos_similarity = 1 - cos_distance
                    
                    # Gabungkan skor dengan weighting
                    # Berikan bobot lebih tinggi ke cross-encoder (0.7) vs cosine (0.3)
                    combined_score = (0.7 * ce_score_normalized) + (0.3 * cos_similarity)
                    
                    combined_scores.append({
                        'index': i,
                        'document': doc,
                        'metadata': metadata,
                        'cosine_similarity': cos_similarity,
                        'cross_encoder_score': ce_score_normalized,
                        'combined_score': combined_score,
                        'original_distance': cos_distance
                    })
                
                # Urutkan berdasarkan combined score
                combined_scores.sort(key=lambda x: x['combined_score'], reverse=True)
                
                # Ambil n_results terbaik
                top_results = combined_scores[:n_results]
                
                # Format hasil
                formatted_results = []
                for i, result in enumerate(top_results):
                    similarity_percent = result['combined_score'] * 100
                    
                    formatted_results.append({
                        'id': result['index'],
                        'rank': i + 1,
                        'document': result['document'],
                        'metadata': result['metadata'],
                        'similarity_score': round(result['combined_score'], 4),
                        'similarity_percent': round(similarity_percent, 2),
                        'cosine_similarity': round(result['cosine_similarity'], 4),
                        'cross_encoder_score': round(result['cross_encoder_score'], 4),
                        'chunk_source': result['metadata'].get('source', 'Unknown') if result['metadata'] else 'Unknown',
                        'chunk_id': result['metadata'].get('chunk_id', result['index']) if result['metadata'] else result['index'],
                        'reranked': True
                    })
                
                logger.info(f"Reranking selesai. Skor tertinggi: {formatted_results[0]['similarity_percent']}%")
                
            else:
                # Fallback: gunakan cosine similarity saja
                logger.info("Menggunakan cosine similarity tanpa reranking")
                formatted_results = []
                
                for i, (doc, metadata, distance) in enumerate(zip(documents[:n_results], 
                                                                  metadatas[:n_results], 
                                                                  distances[:n_results])):
                    similarity_score = 1 - distance
                    similarity_percent = similarity_score * 100
                    
                    formatted_results.append({
                        'id': i,
                        'rank': i + 1,
                        'document': doc,
                        'metadata': metadata,
                        'similarity_score': round(similarity_score, 4),
                        'similarity_percent': round(similarity_percent, 2),
                        'cosine_similarity': round(similarity_score, 4),
                        'cross_encoder_score': 0.0,
                        'chunk_source': metadata.get('source', 'Unknown') if metadata else 'Unknown',
                        'chunk_id': metadata.get('chunk_id', i) if metadata else i,
                        'reranked': False
                    })
            
            # Step 3: Ekstrak istilah penting dari hasil terbaik
            if formatted_results:
                documents = [result['document'] for result in formatted_results]
                self.key_terms = self._extract_key_terms_from_documents(documents)
                logger.info(f"Ekstrak {len(self.key_terms)} istilah penting dari {len(formatted_results)} dokumen")
                
                # Log hasil
                logger.info(f"Ditemukan {len(formatted_results)} dokumen relevan:")
                for result in formatted_results:
                    method = "Cross-Encoder" if result['reranked'] else "Cosine"
                    logger.info(f"  Dokumen {result['rank']}: {result['similarity_percent']}% ({method})")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error dalam similarity search: {e}")
            return []
    
    # ============================================================================
    # FUNGSI EKSTRAKSI ISTILAH PENTING
    # ============================================================================
    
    def _extract_key_terms_from_documents(self, documents: List[str]) -> set:
        """Ekstrak istilah penting dari dokumen"""
        key_terms = set()
        
        for doc in documents:
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', doc)
            acronyms = re.findall(r'\b[A-Z]{2,}\b', doc)
            quoted_terms = re.findall(r'["\'`]([^"\']+)["\'`]', doc)
            
            key_terms.update(proper_nouns)
            key_terms.update(acronyms)
            key_terms.update(quoted_terms)
            
            words = doc.split()
            for word in words:
                if re.search(r'\d', word) and len(word) > 3:
                    key_terms.add(word)
                if re.search(r'[_\-]', word):
                    key_terms.add(word)
        
        return key_terms
    
    def _create_term_protection_patterns(self, key_terms: set) -> List[Tuple[str, str]]:
        """Buat pola regex untuk melindungi istilah penting"""
        patterns = []
        
        for term in sorted(key_terms, key=len, reverse=True):
            if len(term) > 2:
                escaped_term = re.escape(term)
                patterns.append((fr'\b{escaped_term}\b', f'||{term}||'))
        
        return patterns
    
    # ============================================================================
    # FUNGSI PROTECTION DAN POST-PROCESSING ISTILAH
    # ============================================================================
    
    def _protect_key_terms(self, text: str) -> str:
        """Lindungi istilah penting dalam teks dengan penanda khusus"""
        if not self.key_terms:
            return text
        
        protected_text = text
        patterns = self._create_term_protection_patterns(self.key_terms)
        
        for pattern, replacement in patterns:
            protected_text = re.sub(pattern, replacement, protected_text)
        
        return protected_text
    
    def _restore_key_terms(self, text: str) -> str:
        """Kembalikan istilah penting yang sudah diproteksi"""
        restored_text = re.sub(r'\|\|([^|]+)\|\|', r'\1', text)
        return restored_text
    
    # ============================================================================
    # FUNGSI GENERATE RESPONSE
    # ============================================================================
    
    def _format_context_with_structure(self, search_results: List[Dict[str, Any]]) -> str:
        """Format semua hasil similarity search dengan struktur yang jelas"""
        if not search_results:
            return "Tidak ada informasi yang ditemukan."
        
        context_parts = []
        
        for result in search_results:
            rank = result['rank']
            similarity = result['similarity_percent']
            document = result['document']
            source = result['chunk_source']
            chunk_id = result['chunk_id']
            method = "Cross-Encoder" if result.get('reranked', False) else "Cosine"
            
            context_part = f"""
📌 **POIN {rank}** (Kemiripan: {similarity}% | Metode: {method} | Sumber: {source} | Chunk ID: {chunk_id})
{document}
"""
            context_parts.append(context_part)
        
        header = f"📊 **DITEMUKAN {len(search_results)} INFORMASI RELEVAN:**\n\n"
        return header + "\n" + "─" * 50 + "\n".join(context_parts)
    
    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """
        Generate response yang natural dengan menggabungkan konteks.
        """
        # 1. Gabungkan semua konteks menjadi satu teks narasi
        context_text = ""
        for res in search_results:
            context_text += f"{res['document']}\n\n"
        
        # 2. PROMPT YANG DIPERBAIKI (Prompt Engineering)
        # Instruksi ini memaksa AI menjadi konsultan pertanian, bukan pembaca data.
        prompt = f"""<|im_start|>system
Anda adalah Asisten Ahli Pertanian yang cerdas dan membantu. 
Tugas Anda adalah menjawab pertanyaan petani berdasarkan data referensi yang diberikan.

ATURAN PENTING:
1. JANGAN menjawab per poin (seperti "Berdasarkan poin 1..."). 
2. Gabungkan (sintesis) semua informasi menjadi satu penjelasan yang utuh, mengalir, dan mudah dibaca.
3. Jika informasi ada yang berulang, ambil yang paling lengkap.
4. Gunakan Bahasa Indonesia yang natural dan sopan.
5. Jika dokumen membahas "Penyakit" padahal user tanya "Hama", jelaskan perbedaannya.
6. Jika tidak ada jawaban di referensi, katakan: "Maaf, informasi spesifik tentang itu belum ada di database saya," lalu berikan info terdekat yang tersedia.

REFERENSI:
{context_text}
<|im_end|>
<|im_start|>user
Pertanyaan: {query}
<|im_end|>
<|im_start|>assistant
"""
        
        # 3. Generate dengan parameter yang lebih 'kreatif' sedikit
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.3,      # Sedikit naikkan agar luwes, tapi tetap faktual
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True        # Penting untuk hasil yang lebih natural
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Bersihkan sisa prompt dari output (jika ada)
        if "<|im_start|>assistant" in response:
            response = response.split("<|im_start|>assistant")[-1].strip()
        elif "Pertanyaan:" in response:
             response = response.split("assistant\n")[-1].strip()

        return response
    
    # ============================================================================
    # FUNGSI FALLBACK
    # ============================================================================
    
    def _create_comprehensive_fallback(self, search_results: List[Dict[str, Any]]) -> str:
        """Fallback response yang menggabungkan SEMUA informasi"""
        if not search_results:
            return "Maaf, saya tidak menemukan informasi yang relevan dalam knowledge base."
        
        response_parts = []
        response_parts.append(f"📊 **Berdasarkan knowledge base, ditemukan {len(search_results)} informasi relevan:**\n")
        
        for i, result in enumerate(search_results, 1):
            doc = result['document']
            similarity = result['similarity_percent']
            source = result['chunk_source']
            method = "Cross-Encoder" if result.get('reranked', False) else "Cosine"
            
            if len(doc) > 300:
                doc = doc[:300] + "..."
            
            response_part = f"""
{i}. **{source}** (Kemiripan: {similarity}% | Metode: {method})
   {doc}
"""
            response_parts.append(response_part)
        
        response_parts.append("\nℹ️ **Informasi di atas dikumpulkan dari berbagai chunk yang relevan dengan pertanyaan Anda.**")
        
        return "\n".join(response_parts)
    
    # ============================================================================
    # FUNGSI SOCIAL / CONVERSATIONAL CHAT (BARU)
    # ============================================================================

    def _is_social_query(self, query: str) -> bool:
        """Cek apakah query bersifat sosial/sapaan ringan"""
        # Daftar kata kunci sapaan
        social_keywords = [
            "halo", "hello", "hi", "hai", "pagi", "siang", "sore", "malam",
            "apa kabar", "siapa kamu", "terima kasih", "makasih", "thanks",
            "ok", "oke", "baik", "siap", "test", "tes", "permisi", "assalamualaikum"
        ]
        
        query_lower = query.lower()
        
        # Cek 1: Jika query sangat pendek (< 2 kata) dan bukan angka
        if len(query.split()) < 2 and not any(char.isdigit() for char in query):
            return True
            
        # Cek 2: Jika mengandung kata kunci sosial
        # Kita cek apakah salah satu keyword ada di awal kalimat atau berdiri sendiri
        for word in social_keywords:
            # Regex untuk mencocokkan kata utuh (bukan bagian dari kata lain)
            if re.search(fr'\b{word}\b', query_lower):
                return True
                
        return False

    def _generate_social_response(self, query: str) -> str:
        """Generate balasan santai menggunakan Chat Template (Agar tidak halusinasi)"""
        try:
            # 1. Gunakan format pesan standar (System & User)
            messages = [
                {"role": "system", "content": "Anda adalah asisten AI yang ramah, sopan, dan membantu. Jawab sapaan user dengan singkat (maksimal 2 kalimat) dalam Bahasa Indonesia yang natural. Jangan membuat percakapan fiktif sendiri."},
                {"role": "user", "content": query}
            ]

            # 2. Format menggunakan template bawaan model (PENTING untuk Qwen/Llama)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=100,
                    do_sample=True,
                    temperature=0.6,       # Cukup kreatif tapi tidak liar
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 3. Decode HANYA bagian jawaban baru (membuang prompt input)
            # Ini mencegah teks "User:..." atau "System:..." muncul di jawaban
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error social response: {e}")
            return "Halo! Ada yang bisa saya bantu terkait dokumen Anda hari ini?"
    
    # ============================================================================
    # FUNGSI UTAMA CHAT
    # ============================================================================
    
    def chat(self, query: str, safe_mode: bool = True, n_results: int = 3, 
             correct_typos: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        """
        Fungsi utama chat dengan fitur deteksi Social Chat (FIXED METADATA)
        """
        logger.info(f"Memproses query: {query}")
        
        original_query = query
        corrected_query = query
        
        # 1. Koreksi Typo (Hanya jika bukan kata pendek/sapaan)
        if correct_typos:
            # Cek dulu apakah ini kemungkinan sapaan pendek agar tidak "over-correct"
            if len(query.split()) >= 2: 
                corrected_query = self.correct_typos(query)
            else:
                # Jika cuma 1 kata, biarkan (misal: "Halo" jangan jadi "Hal")
                pass

        # -----------------------------------------------------------
        # 2. LOGIKA SOCIAL CHAT (Dengan Metadata Lengkap)
        # -----------------------------------------------------------
        if self._is_social_query(original_query):
            logger.info("Terdeteksi sebagai Social Chat. Bypass RAG.")
            response = self._generate_social_response(original_query)
            
            # PERBAIKAN DI SINI: Menyertakan semua key yang dibutuhkan main()
            return {
                'response': response,
                'metadata': {
                    'query': original_query,
                    'corrected_query': original_query, # Social chat tidak perlu koreksi
                    'type': 'social_chat',
                    'total_results': 0,
                    # --- Key di bawah ini WAJIB ada agar main() tidak error ---
                    'typo_correction_applied': False, 
                    'use_reranking': False,
                    'n_results_requested': 0,
                    'similarity_scores': [],
                    'key_terms_count': 0
                },
                'search_results': []
            }
        # -----------------------------------------------------------

        # 3. Jika bukan social chat, lanjut ke proses RAG biasa
        if correct_typos and corrected_query != original_query:
             logger.info(f"Query dikoreksi: '{original_query}' -> '{corrected_query}'")

        # Lakukan similarity search
        if use_reranking and self.cross_encoder is not None:
            logger.info("Menggunakan cross-encoder reranking")
            search_results = self.similarity_search(
                corrected_query, 
                n_results=n_results,
                initial_candidates=10,
                rerank_top_k=10
            )
        else:
            logger.info("Menggunakan cosine similarity tanpa reranking")
            search_results = self.similarity_search(corrected_query, n_results=n_results)
        
        # Generate response RAG
        response = self.generate_response(corrected_query, search_results)
        
        # Metadata standar untuk Search
        metadata = {
            'query': original_query,
            'corrected_query': corrected_query if correct_typos else original_query,
            'type': 'knowledge_search',
            'typo_correction_applied': correct_typos and (corrected_query != original_query),
            'total_results': len(search_results),
            'key_terms_count': len(self.key_terms),
            'key_terms_sample': list(self.key_terms)[:10] if self.key_terms else [],
            'safe_mode': safe_mode,
            'n_results_requested': n_results,
            'use_reranking': use_reranking and self.cross_encoder is not None,
            'similarity_scores': [result['similarity_percent'] for result in search_results]
        }
        
        return {
            'response': response,
            'metadata': metadata,
            'search_results': search_results
        }
    
    # ============================================================================
    # FUNGSI VALIDASI PRESERVATION ISTILAH
    # ============================================================================
    
    def _validate_term_preservation(self, response: str, original_docs: List[str]):
        """Validasi apakah istilah penting tetap terjaga dalam response"""
        try:
            all_original_terms = set()
            for doc in original_docs:
                terms = self._extract_key_terms_from_documents([doc])
                all_original_terms.update(terms)
            
            if not all_original_terms:
                return
            
            missing_terms = []
            preserved_terms = []
            
            for term in all_original_terms:
                if len(term) > 3:
                    pattern = fr'\b{re.escape(term)}\b'
                    if re.search(pattern, response, re.IGNORECASE):
                        preserved_terms.append(term)
                    else:
                        variations = [
                            term.replace(' ', '_'),
                            term.replace(' ', '-'),
                            term.lower(),
                            term.upper()
                        ]
                        
                        found = False
                        for variation in variations:
                            if variation in response:
                                preserved_terms.append(term)
                                found = True
                                break
                        
                        if not found:
                            missing_terms.append(term)
            
            if missing_terms:
                logger.warning(f"{len(missing_terms)} istilah mungkin tidak muncul: {missing_terms[:5]}")
            else:
                logger.info(f"Validasi istilah: {len(preserved_terms)} dari {len(all_original_terms)} istilah penting terjaga")
                
        except Exception as e:
            logger.error(f"Error dalam validasi: {e}")
    
    # ============================================================================
    # FUNGSI UTILITAS
    # ============================================================================
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Mendapatkan informasi tentang koleksi"""
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'total_documents': count,
                'chroma_dir': self.chroma_dir,
                'cross_encoder_available': self.cross_encoder is not None
            }
        except Exception as e:
            logger.error(f"Error mendapatkan info koleksi: {e}")
            return {}
    
    def clear_memory(self):
        """Clear GPU memory jika menggunakan GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# FUNGSI MAIN DAN INTERFACE
# ============================================================================

def main():
    """Fungsi utama untuk menjalankan chatbot"""
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "knowledge_base_new2"
    MODEL_PATH = "./model/Qwen2.5-3B-Instruct"
    CROSS_ENCODER_MODEL = "unicamp-dl/mmarco-mMiniLM-v6-translated-gpu"  # Model cross-encoder yang efisien
    
    try:
        print("🔄 Menginisialisasi chatbot dengan cross-encoder...")
        chatbot = KnowledgeBaseChatbot(
            chroma_dir=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            model_path=MODEL_PATH,
            cross_encoder_model=CROSS_ENCODER_MODEL
        )
        
        info = chatbot.get_collection_info()
        print(f"📚 Knowledge Base: {info['collection_name']}")
        print(f"📄 Total dokumen: {info['total_documents']}")
        print(f"🔤 Typo Correction: Aktif")
        print(f"🎯 Cross-Encoder: {'Aktif' if info['cross_encoder_available'] else 'Tidak aktif'}")
        print("\n" + "="*60)
        print("🤖 Chatbot siap! Ketik 'exit' untuk keluar")
        print("✅ Mode: Menggabungkan SEMUA informasi relevan")
        print("✅ Fitur: Cross-Encoder reranking untuk hasil lebih akurat")
        print("✅ Fitur: Koreksi typo otomatis")
        print("="*60 + "\n")
        
        while True:
            user_query = input("👤 Anda: ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'keluar']:
                print("👋 Terima kasih, sampai jumpa!")
                break
            
            if not user_query:
                print("⚠️  Silakan masukkan pertanyaan\n")
                continue
            
            # Konfigurasi interaktif
            print("\n🔧 **KONFIGURASI SEARCH**")
            print("🔤 Aktifkan typo correction? [1] Ya [2] Tidak")
            typo_choice = input("Pilihan (1/2): ").strip()
            correct_typos = typo_choice != "2"
            
            print("🎯 Gunakan cross-encoder reranking? [1] Ya [2] Tidak")
            rerank_choice = input("Pilihan (1/2): ").strip()
            use_reranking = rerank_choice != "2"
            
            print("🔢 Berapa banyak hasil yang ingin diambil? (default: 5)")
            n_results_input = input("Jumlah (3-20): ").strip()
            try:
                n_results = int(n_results_input) if n_results_input else 5
                n_results = max(3, min(n_results, 20))
            except:
                n_results = 5
            
            # Proses query
            print(f"\n🔍 Mencari informasi dengan konfigurasi:")
            print(f"   • Typo correction: {'Aktif' if correct_typos else 'Tidak aktif'}")
            print(f"   • Cross-encoder: {'Aktif' if use_reranking else 'Tidak aktif'}")
            print(f"   • Jumlah hasil: {n_results}")
            
            result = chatbot.chat(
                user_query, 
                safe_mode=True, 
                n_results=n_results,
                correct_typos=correct_typos,
                use_reranking=use_reranking
            )
            
            if correct_typos and result['metadata']['typo_correction_applied']:
                print(f"🔤 Query dikoreksi: '{result['metadata']['query']}' -> '{result['metadata']['corrected_query']}'")
            
            print(f"\n{'='*60}")
            print("🤖 ASSISTANT:")
            print(f"{'='*60}")
            print(f"{result['response']}\n")
            
            # Tampilkan metadata (opsional)
            debug_mode = False  # Set ke True untuk debugging
            if debug_mode:
                print(f"{'='*60}")
                print("📊 METADATA SEARCH:")
                print(f"{'='*60}")
                print(f"• Query asli: {result['metadata']['query']}")
                if result['metadata']['typo_correction_applied']:
                    print(f"• Query dikoreksi: {result['metadata']['corrected_query']}")
                print(f"• Total hasil ditemukan: {result['metadata']['total_results']}")
                print(f"• Cross-encoder reranking: {'Ya' if result['metadata']['use_reranking'] else 'Tidak'}")
                print(f"• Jumlah hasil diminta: {result['metadata']['n_results_requested']}")
                
                if result['metadata']['total_results'] > 0:
                    print(f"• Skor similarity: {result['metadata']['similarity_scores']}")
                    print(f"• Istilah penting ditemukan: {result['metadata']['key_terms_count']}")
                
                if result.get('search_results'):
                    print(f"\n📋 TOP 3 HASIL SEARCH:")
                    for i, search_result in enumerate(result['search_results'][:3], 1):
                        method = "Cross-Encoder" if search_result.get('reranked', False) else "Cosine"
                        print(f"  {i}. Similarity: {search_result['similarity_percent']}% ({method}) | Sumber: {search_result['chunk_source']}")
                
                print()
            
            chatbot.clear_memory()
            
    except Exception as e:
        print(f"❌ Error: {e}")
        logging.error(f"Error dalam main function: {e}")


if __name__ == "__main__":
    main()