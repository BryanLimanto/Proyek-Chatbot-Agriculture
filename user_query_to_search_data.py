import os
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import re
from symspellpy import SymSpell, Verbosity
from sentence_transformers import CrossEncoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseChatbot:
    def __init__(self, chroma_dir: str, collection_name: str, model_path: str, 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.chroma_dir = chroma_dir
        self.collection_name = collection_name
        self.model_path = model_path
        self.cross_encoder_model = cross_encoder_model
        
        self._init_chroma_client()
        self._load_llm_model()
        self._init_symspell()
        self._load_cross_encoder()
        
        self.key_terms = set()
        logger.info("Chatbot berhasil diinisialisasi")
    
    def _load_cross_encoder(self):
        try:
            logger.info(f"Memuat cross-encoder: {self.cross_encoder_model}")
            self.cross_encoder = CrossEncoder(self.cross_encoder_model)
            logger.info("Cross-encoder berhasil dimuat")
        except Exception as e:
            logger.error(f"Error memuat cross-encoder: {e}")
            self.cross_encoder = None
    
    def _init_symspell(self):
        try:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dictionary_path = "./dataset/indonesian_dictionary.txt"
            if os.path.exists(dictionary_path):
                self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
                logger.info(f"Kamus dimuat dari {dictionary_path}")
            else:
                self._create_default_dictionary()
                logger.info("Kamus default dibuat")
        except Exception as e:
            logger.error(f"Error inisialisasi SymSpell: {e}")
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    
    def _create_default_dictionary(self):
        words = [
            "apa", "bagaimana", "dimana", "kapan", "siapa", "mengapa",
            "hama", "kentang", "jagung", "cabai", "tanaman", "serangga",
            "penyakit", "organisme", "populasi", "spesies", "genus"
        ]
        for word in words:
            self.sym_spell.create_dictionary_entry(word, 1)
    
    def correct_typos(self, text: str, max_edit_distance: int = 2) -> str:
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
                
                if suggestions and suggestions[0].distance <= max_edit_distance:
                    corrected_words.append(suggestions[0].term)
                else:
                    corrected_words.append(word)
            
            return " ".join(corrected_words)
        except Exception as e:
            logger.error(f"Error koreksi typo: {e}")
            return text
    
    def _init_chroma_client(self):
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
    
    def _load_llm_model(self):
        try:
            logger.info(f"Memuat model LLM dari: {self.model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True
            )
            
            if not torch.cuda.is_available():
                self.model = self.model.to('cpu')
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Model LLM berhasil dimuat")
        except Exception as e:
            logger.error(f"Error memuat model LLM: {e}")
            raise
    
    # ===== FUNGSI SIMILARITY SEARCH DIPERBAIKI =====
    def similarity_search(self, query: str, n_results: int = 5, 
                         initial_candidates: int = 15) -> List[Dict[str, Any]]:
        """
        Similarity search dengan cross-encoder reranking (FIXED VERSION)
        """
        try:
            logger.info(f"Mengambil {initial_candidates} kandidat dari ChromaDB")
            initial_results = self.collection.query(
                query_texts=[query],
                n_results=min(initial_candidates, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            if not initial_results['documents'] or not initial_results['documents'][0]:
                logger.warning("Tidak ada dokumen ditemukan")
                return []
            
            documents = initial_results['documents'][0]
            metadatas = initial_results['metadatas'][0]
            distances = initial_results['distances'][0]
            
            # Rerank dengan cross-encoder
            if self.cross_encoder is not None and len(documents) > 1:
                logger.info(f"Reranking {len(documents)} kandidat dengan cross-encoder")
                
                pairs = [(query, doc) for doc in documents]
                cross_encoder_scores = self.cross_encoder.predict(pairs)
                
                # PERBAIKAN: Hitung combined score dengan benar
                results_with_scores = []
                for i, (doc, metadata, cos_distance, ce_score) in enumerate(
                    zip(documents, metadatas, distances, cross_encoder_scores)
                ):
                    # Normalisasi skor
                    cos_similarity = max(0, min(1, 1 - cos_distance))  # Clamp ke [0,1]
                    ce_score_normalized = max(0, min(1, float(ce_score)))  # Clamp ke [0,1]
                    
                    # Combined score (weighted average)
                    combined_score = (0.7 * ce_score_normalized) + (0.3 * cos_similarity)
                    combined_score = max(0, min(1, combined_score))  # Pastikan [0,1]
                    
                    results_with_scores.append({
                        'index': i,
                        'document': doc,
                        'metadata': metadata,
                        'cosine_similarity': cos_similarity,
                        'cross_encoder_score': ce_score_normalized,
                        'combined_score': combined_score
                    })
                
                # Sort berdasarkan combined score
                results_with_scores.sort(key=lambda x: x['combined_score'], reverse=True)
                
                # Ambil top N results
                top_results = results_with_scores[:n_results]
                
                # Format hasil
                formatted_results = []
                for rank, result in enumerate(top_results, 1):
                    formatted_results.append({
                        'rank': rank,
                        'document': result['document'],
                        'metadata': result['metadata'],
                        'similarity_score': round(result['combined_score'], 4),
                        'similarity_percent': round(result['combined_score'] * 100, 2),  # FIX: 0-100%
                        'cosine_similarity': round(result['cosine_similarity'], 4),
                        'cross_encoder_score': round(result['cross_encoder_score'], 4),
                        'chunk_source': result['metadata'].get('source', 'Unknown') if result['metadata'] else 'Unknown',
                        'reranked': True
                    })
                
                logger.info(f"Reranking selesai. Top score: {formatted_results[0]['similarity_percent']}%")
                
            else:
                # Fallback: cosine similarity only
                formatted_results = []
                for i, (doc, metadata, distance) in enumerate(zip(documents[:n_results], 
                                                                  metadatas[:n_results], 
                                                                  distances[:n_results])):
                    similarity_score = max(0, min(1, 1 - distance))
                    
                    formatted_results.append({
                        'rank': i + 1,
                        'document': doc,
                        'metadata': metadata,
                        'similarity_score': round(similarity_score, 4),
                        'similarity_percent': round(similarity_score * 100, 2),
                        'cosine_similarity': round(similarity_score, 4),
                        'cross_encoder_score': 0.0,
                        'chunk_source': metadata.get('source', 'Unknown') if metadata else 'Unknown',
                        'reranked': False
                    })
            
            # Ekstrak key terms
            if formatted_results:
                docs = [r['document'] for r in formatted_results]
                self.key_terms = self._extract_key_terms_from_documents(docs)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error dalam similarity search: {e}")
            return []
    
    def _extract_key_terms_from_documents(self, documents: List[str]) -> set:
        """Ekstrak istilah penting dari dokumen"""
        key_terms = set()
        
        for doc in documents:
            # Ekstrak proper nouns (huruf kapital)
            proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', doc)
            key_terms.update(proper_nouns)
            
            # Ekstrak istilah ilmiah (mengandung sp., spp., dll)
            scientific_terms = re.findall(r'\b\w+\s+sp\.?\b|\b\w+\s+spp\.?\b', doc, re.IGNORECASE)
            key_terms.update(scientific_terms)
            
            # Ekstrak kata dengan huruf kapital semua (akronim)
            acronyms = re.findall(r'\b[A-Z]{2,}\b', doc)
            key_terms.update(acronyms)
        
        return key_terms
    
    # ===== PROMPT DIPERBAIKI (LEBIH SEDERHANA) =====
    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Generate response dengan prompt yang lebih sederhana dan efektif"""
        try:
            if not search_results:
                return "Maaf, saya tidak menemukan informasi relevan dalam knowledge base."
            
            # Format context dengan JELAS
            context_parts = []
            for i, result in enumerate(search_results, 1):
                doc = result['document']
                score = result['similarity_percent']
                context_parts.append(f"[Dokumen {i}] (Relevansi: {score}%)\n{doc}\n")
            
            full_context = "\n".join(context_parts)
            
            # PROMPT BARU: Sederhana, jelas, tanpa emoji berlebihan
            prompt = f"""Anda adalah asisten yang menjawab pertanyaan tentang hama tanaman berdasarkan knowledge base.

KONTEKS DARI KNOWLEDGE BASE:
{full_context}

PERTANYAAN: {query}

INSTRUKSI:
1. Jawab pertanyaan berdasarkan SEMUA informasi di atas
2. Jika ditanya tentang "hama [tanaman]", sebutkan:
   - Nama-nama spesies/genus hama (misal: Empoasca sp., Liriomyza sp.)
   - Ciri-ciri atau gejala serangan
   - Data populasi jika ada
3. Gunakan NAMA ILMIAH yang persis seperti di dokumen
4. Jangan hanya menyebutkan ordo/famili, sebutkan nama spesifik
5. Jika informasi tidak lengkap, katakan "Berdasarkan knowledge base yang tersedia..."
6. Gunakan bahasa Indonesia yang jelas

JAWABAN:"""
            
            # Tokenize dan generate
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=3000,  # Kurangi untuk model 3B
                padding=True
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.to('cuda') for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=300,  # Kurangi output length
                    temperature=0.1,  # Lebih deterministik
                    do_sample=False,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(prompt, "").strip()
            
            # Jika response terlalu pendek atau tidak menjawab, gunakan fallback
            if len(response) < 50 or "Berdasarkan knowledge base" not in response:
                logger.warning("Response terlalu pendek, menggunakan fallback")
                return self._create_fallback_response(query, search_results)
            
            return response
            
        except Exception as e:
            logger.error(f"Error generate response: {e}")
            return self._create_fallback_response(query, search_results)
    
    def _create_fallback_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Fallback response langsung dari dokumen"""
        if not search_results:
            return "Maaf, tidak ada informasi yang ditemukan."
        
        # Ekstrak nama-nama hama dari dokumen
        hama_list = set()
        for result in search_results:
            doc = result['document']
            # Cari pola seperti "Empoasca sp.", "Liriomyza sp.", dll
            matches = re.findall(r'\b[A-Z][a-z]+\s+sp\.?\b', doc)
            hama_list.update(matches)
            
            # Cari pola nama ilmiah lengkap
            matches2 = re.findall(r'\b[A-Z][a-z]+\s+[a-z]+\b', doc)
            hama_list.update(matches2)
        
        if not hama_list:
            # Jika tidak ada nama ilmiah, berikan ringkasan dokumen
            response = f"Berdasarkan knowledge base tentang '{query}':\n\n"
            for i, result in enumerate(search_results[:3], 1):
                doc_snippet = result['document'][:200] + "..."
                response += f"{i}. {doc_snippet}\n\n"
            return response
        
        # Format jawaban dengan daftar hama
        response = f"Berdasarkan knowledge base, hama yang ditemukan pada {query.replace('hama', '').strip()}:\n\n"
        for i, hama in enumerate(sorted(hama_list), 1):
            response += f"{i}. {hama}\n"
        
        response += f"\n(Total: {len(hama_list)} jenis hama ditemukan dalam knowledge base)"
        
        return response
    
    # ===== FUNGSI CHAT UTAMA (SIMPLIFIED) =====
    def chat(self, query: str, n_results: int = 5, 
             correct_typos: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        """Chat dengan konfigurasi optimal"""
        logger.info(f"Query: {query}")
        
        original_query = query
        if correct_typos:
            query = self.correct_typos(query)
        
        # Search dengan parameter optimal untuk model kecil
        search_results = self.similarity_search(
            query, 
            n_results=min(n_results, 5),  # Maksimal 5 untuk model 3B
            initial_candidates=10
        )
        
        # Generate response
        response = self.generate_response(query, search_results)
        
        metadata = {
            'query': original_query,
            'corrected_query': query,
            'total_results': len(search_results),
            'top_score': search_results[0]['similarity_percent'] if search_results else 0
        }
        
        return {
            'response': response,
            'metadata': metadata,
            'search_results': search_results
        }
    
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            return {
                'collection_name': self.collection_name,
                'total_documents': self.collection.count(),
                'cross_encoder_available': self.cross_encoder is not None
            }
        except Exception as e:
            logger.error(f"Error: {e}")
            return {}
    
    def clear_memory(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ===== MAIN FUNCTION =====
def main():
    CHROMA_DIR = "./chroma_db"
    COLLECTION_NAME = "test3"
    MODEL_PATH = "./model/Qwen2.5-3B-Instruct"
    
    try:
        print("🔄 Menginisialisasi chatbot (versi diperbaiki)...")
        chatbot = KnowledgeBaseChatbot(
            chroma_dir=CHROMA_DIR,
            collection_name=COLLECTION_NAME,
            model_path=MODEL_PATH
        )
        
        info = chatbot.get_collection_info()
        print(f"📚 Knowledge Base: {info['collection_name']}")
        print(f"📄 Total dokumen: {info['total_documents']}")
        print(f"🎯 Cross-Encoder: {'Aktif' if info['cross_encoder_available'] else 'Tidak'}")
        print("\n" + "="*60)
        print("🤖 Chatbot siap! (versi diperbaiki)")
        print("="*60 + "\n")
        
        while True:
            user_query = input("👤 Anda: ").strip()
            
            if user_query.lower() in ['exit', 'quit', 'keluar']:
                print("👋 Terima kasih!")
                break
            
            if not user_query:
                continue
            
            # Konfigurasi sederhana
            print("\n🔍 Mencari dan memproses...")
            
            result = chatbot.chat(
                user_query, 
                n_results=5,  # Optimal untuk model 3B
                correct_typos=True,
                use_reranking=True
            )
            
            print(f"\n{'='*60}")
            print("🤖 JAWABAN:")
            print(f"{'='*60}")
            print(f"{result['response']}\n")
            
            # Info singkat
            print(f"ℹ️  Ditemukan {result['metadata']['total_results']} dokumen relevan")
            if result['metadata']['total_results'] > 0:
                print(f"ℹ️  Skor tertinggi: {result['metadata']['top_score']}%\n")
            
            chatbot.clear_memory()
            
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()