import os
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging
import re
from symspellpy import SymSpell, Verbosity
from sentence_transformers import CrossEncoder

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

        # 1. Embedding Function (Wajib sama dengan saat Ingest)
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
            logger.info(f"ChromaDB collection '{self.collection_name}' diakses")
        except Exception as e:
            logger.error(f"Error inisialisasi ChromaDB: {e}")
            raise

    def _load_cross_encoder(self):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.cross_encoder = CrossEncoder(self.cross_encoder_model, device=device)
            logger.info(f"Cross-encoder dimuat di {device}")
        except Exception as e:
            logger.error(f"Gagal memuat Cross-Encoder: {e}")
            self.cross_encoder = None

    def _init_symspell(self):
        try:
            self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            dictionary_path = "./dataset/indonesian_dictionary.txt"
            if os.path.exists(dictionary_path):
                self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
            else:
                self._create_default_dictionary()
        except Exception as e:
            logger.error(f"Error SymSpell: {e}")

    def _create_default_dictionary(self):
        words = ["halo", "apa", "bagaimana", "hama", "penyakit", "tanaman", "padi", "pupuk", "obat"]
        for word in words:
            self.sym_spell.create_dictionary_entry(word, 1)

    def correct_typos(self, text: str) -> str:
        try:
            words = text.split()
            corrected_words = [
                self.sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True)[0].term 
                if len(w) > 3 else w for w in words
            ]
            return " ".join(corrected_words)
        except:
            return text

    def _load_llm_model(self):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Error load LLM: {e}")
            raise

    def similarity_search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        try:
            initial_results = self.collection.query(
                query_texts=[query],
                n_results=min(10, self.collection.count()),
                include=["documents", "metadatas", "distances"]
            )
            
            if not initial_results['documents'][0]: return []

            docs, metas, dists = initial_results['documents'][0], initial_results['metadatas'][0], initial_results['distances'][0]
            
            # Reranking Logic
            if self.cross_encoder and len(docs) > 1:
                pairs = [(query, doc) for doc in docs]
                ce_scores = self.cross_encoder.predict(pairs)
                
                results = []
                for i in range(len(docs)):
                    sim = 1 - dists[i]
                    # Gabungan skor (70% Cross-Encoder, 30% Cosine)
                    combined = (0.7 * float(ce_scores[i])) + (0.3 * sim)
                    results.append({
                        'rank': 0, 'document': docs[i], 'metadata': metas[i],
                        'similarity_percent': round(combined * 100, 2),
                        'reranked': True, 'chunk_source': metas[i].get('source', 'Unknown'),
                        'chunk_id': metas[i].get('chunk_id', i), 'score': combined
                    })
                results.sort(key=lambda x: x['score'], reverse=True)
                for i, r in enumerate(results): r['rank'] = i + 1
                return results[:n_results]
            
            return [{'rank': i+1, 'document': d, 'similarity_percent': round((1-dist)*100, 2), 
                     'chunk_source': m.get('source', 'Unknown'), 'chunk_id': m.get('chunk_id', i), 'reranked': False} 
                    for i, (d, m, dist) in enumerate(zip(docs[:n_results], metas[:n_results], dists[:n_results]))]
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

    def generate_response(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        context = "\n\n".join([r['document'] for r in search_results])
        prompt = f"""<|im_start|>system
Anda adalah Asisten Ahli Pertanian. Jawablah secara mengalir, sopan, dan edukatif dalam Bahasa Indonesia.
REFERENSI: {context}<|im_end|>
<|im_start|>user
{query}<|im_end|>
<|im_start|>assistant
"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=512, temperature=0.3, do_sample=True)
        
        full_res = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return full_res.split("assistant")[-1].strip()

    def _is_social_query(self, query: str) -> bool:
        social_keywords = ["halo", "hi", "hai", "pagi", "siang", "sore", "malam", "siapa kamu", "terima kasih", "makasih"]
        return any(re.search(fr'\b{word}\b', query.lower()) for word in social_keywords) or len(query.split()) < 2

    def _generate_social_response(self, query: str) -> str:
        messages = [{"role": "system", "content": "Anda asisten AI pertanian yang ramah. Jawab sapaan dengan singkat."},
                    {"role": "user", "content": query}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            gen = self.model.generate(inputs.input_ids, max_new_tokens=50, temperature=0.6, do_sample=True)
        return self.tokenizer.decode(gen[0][len(inputs.input_ids[0]):], skip_special_tokens=True).strip()

    def chat(self, query: str, n_results: int = 3, correct_typos: bool = True, use_reranking: bool = True) -> Dict[str, Any]:
        original_query = query
        if self._is_social_query(query):
            return {'response': self._generate_social_response(query), 'metadata': {'type': 'social', 'typo_correction_applied': False}, 'search_results': []}

        corrected = self.correct_typos(query) if correct_typos else query
        search_results = self.similarity_search(corrected, n_results=n_results)
        response = self.generate_response(corrected, search_results)
        
        return {
            'response': response,
            'search_results': search_results,
            'metadata': {
                'type': 'rag',
                'corrected_query': corrected,
                'typo_correction_applied': corrected.lower() != original_query.lower()
            }
        }

    def get_collection_info(self):
        return {'total_documents': self.collection.count(), 'cross_encoder_available': self.cross_encoder is not None}

    def clear_memory(self):
        if torch.cuda.is_available(): torch.cuda.empty_cache()