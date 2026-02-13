import chromadb
from chromadb.utils import embedding_functions

# Konfigurasi
CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "multilingual_e5"

def view_collection_info():
    """Menampilkan informasi dasar collection"""
    print("=" * 50)
    print("CHROMADB VECTOR DATABASE VIEWER")
    print("=" * 50)
    
    # Inisialisasi client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    # Dapatkan semua collections
    collections = client.list_collections()
    print(f"\nTotal Collections: {len(collections)}")
    
    for idx, collection in enumerate(collections, 1):
        print(f"\n{idx}. Collection: '{collection.name}'")
        print(f"   Jumlah dokumen: {collection.count()}")
    
    return collections

def view_collection_details(collection_name=COLLECTION_NAME):
    """Menampilkan detail collection tertentu"""
    print(f"\n{'='*50}")
    print(f"VIEWING COLLECTION: '{collection_name}'")
    print(f"{'='*50}")
    
    # Inisialisasi client
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    
    try:
        # Dapatkan collection
        collection = client.get_collection(name=collection_name)
        
        # Hitung total dokumen
        total_docs = collection.count()
        print(f"\nTotal dokumen/chunks: {total_docs}")
        
        # Ambil semua data
        results = collection.get()
        
        if not results['ids']:
            print("Collection kosong!")
            return
        
        # Tampilkan statistik
        print(f"\n{'='*50}")
        print("STATISTIK DOKUMEN")
        print(f"{'='*50}")
        
        # Hitung dokumen per kategori
        kategori_count = {}
        source_count = {}
        
        for metadata in results['metadatas']:
            kategori = metadata.get('kategori', 'Unknown')
            source = metadata.get('source', 'Unknown')
            
            kategori_count[kategori] = kategori_count.get(kategori, 0) + 1
            source_count[source] = source_count.get(source, 0) + 1
        
        print("\nDokumen per Kategori:")
        for kategori, count in kategori_count.items():
            print(f"  - {kategori}: {count} dokumen")
        
        print("\nDokumen per File Source:")
        for source, count in source_count.items():
            print(f"  - {source}: {count} dokumen")
        
        # Menu untuk melihat detail
        while True:
            print(f"\n{'='*50}")
            print("MENU VIEWER:")
            print("1. Lihat semua dokumen")
            print("2. Lihat dokumen berdasarkan kategori")
            print("3. Lihat dokumen berdasarkan file source")
            print("4. Cari dokumen dengan query")
            print("5. Lihat beberapa contoh dokumen")
            print("6. Keluar")
            
            pilihan = input("\nPilih menu (1-6): ").strip()
            
            if pilihan == "1":
                show_all_documents(collection, results)
            elif pilihan == "2":
                show_documents_by_category(collection, results)
            elif pilihan == "3":
                show_documents_by_source(collection, results)
            elif pilihan == "4":
                search_documents(collection)
            elif pilihan == "5":
                show_sample_documents(results)
            elif pilihan == "6":
                print("Keluar dari program.")
                break
            else:
                print("Pilihan tidak valid!")
                
    except Exception as e:
        print(f"Error: {e}")
        print(f"Collection '{collection_name}' tidak ditemukan!")

def show_all_documents(collection, results):
    """Menampilkan semua dokumen"""
    print(f"\n{'='*50}")
    print("SEMUA DOKUMEN")
    print(f"{'='*50}")
    
    for i, (doc_id, document, metadata) in enumerate(zip(
        results['ids'], 
        results['documents'], 
        results['metadatas']
    ), 1):
        print(f"\n[{i}] ID: {doc_id}")
        print(f"Kategori: {metadata.get('kategori', 'N/A')}")
        print(f"Source: {metadata.get('source', 'N/A')}")
        print(f"Preview: {document[:200]}...")
        
        if i % 5 == 0:
            lanjut = input("\nLanjut? (y/n): ").lower()
            if lanjut != 'y':
                break

def show_documents_by_category(collection, results):
    """Menampilkan dokumen berdasarkan kategori"""
    # Ambil semua kategori unik
    categories = set()
    for metadata in results['metadatas']:
        categories.add(metadata.get('kategori', 'Unknown'))
    
    print("\nKategori yang tersedia:")
    for idx, cat in enumerate(sorted(categories), 1):
        print(f"{idx}. {cat}")
    
    pilihan = input("\nPilih kategori (nama atau angka): ").strip()
    
    # Cari dokumen dengan kategori tersebut
    filtered_docs = []
    for doc_id, document, metadata in zip(
        results['ids'], 
        results['documents'], 
        results['metadatas']
    ):
        if metadata.get('kategori') == pilihan or \
           (pilihan.isdigit() and list(sorted(categories))[int(pilihan)-1] == metadata.get('kategori')):
            filtered_docs.append((doc_id, document, metadata))
    
    if not filtered_docs:
        print("Tidak ada dokumen dengan kategori tersebut!")
        return
    
    print(f"\n{'='*50}")
    print(f"DOkUMEN DENGAN KATEGORI: {filtered_docs[0][2].get('kategori')}")
    print(f"Jumlah: {len(filtered_docs)} dokumen")
    print(f"{'='*50}")
    
    for i, (doc_id, document, metadata) in enumerate(filtered_docs, 1):
        print(f"\n[{i}] ID: {doc_id}")
        print(f"Source: {metadata.get('source', 'N/A')}")
        print(f"Content: {document}")
        print(f"{'-'*30}")
        
        if i % 3 == 0:
            lanjut = input("\nLanjut? (y/n): ").lower()
            if lanjut != 'y':
                break

def show_documents_by_source(collection, results):
    """Menampilkan dokumen berdasarkan file source"""
    # Ambil semua source unik
    sources = set()
    for metadata in results['metadatas']:
        sources.add(metadata.get('source', 'Unknown'))
    
    print("\nFile sources yang tersedia:")
    for idx, src in enumerate(sorted(sources), 1):
        print(f"{idx}. {src}")
    
    pilihan = input("\nPilih file source (nama atau angka): ").strip()
    
    # Cari dokumen dengan source tersebut
    filtered_docs = []
    for doc_id, document, metadata in zip(
        results['ids'], 
        results['documents'], 
        results['metadatas']
    ):
        if metadata.get('source') == pilihan or \
           (pilihan.isdigit() and list(sorted(sources))[int(pilihan)-1] == metadata.get('source')):
            filtered_docs.append((doc_id, document, metadata))
    
    if not filtered_docs:
        print("Tidak ada dokumen dengan source tersebut!")
        return
    
    print(f"\n{'='*50}")
    print(f"DOkUMEN DENGAN SOURCE: {filtered_docs[0][2].get('source')}")
    print(f"Jumlah: {len(filtered_docs)} dokumen")
    print(f"{'='*50}")
    
    for i, (doc_id, document, metadata) in enumerate(filtered_docs, 1):
        print(f"\n[{i}] ID: {doc_id}")
        print(f"Kategori: {metadata.get('kategori', 'N/A')}")
        print(f"Content: {document}")
        print(f"{'-'*30}")
        
        if i % 3 == 0:
            lanjut = input("\nLanjut? (y/n): ").lower()
            if lanjut != 'y':
                break

def search_documents(collection):
    """Mencari dokumen dengan query"""
    query = input("\nMasukkan query pencarian: ").strip()
    
    if not query:
        print("Query tidak boleh kosong!")
        return
    
    try:
        n_results = input("Jumlah hasil (default: 5): ").strip()
        n_results = int(n_results) if n_results.isdigit() else 5
        
        print(f"\nMencari: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, 20)
        )
        
        if not results['ids'][0]:
            print("Tidak ada hasil yang ditemukan!")
            return
        
        print(f"\n{'='*50}")
        print(f"HASIL PENCARIAN: '{query}'")
        print(f"{'='*50}")
        
        for i, (doc_id, document, metadata, distance) in enumerate(zip(
            results['ids'][0],
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        ), 1):
            print(f"\n[{i}] Similarity: {1-distance:.4f}")
            print(f"ID: {doc_id}")
            print(f"Kategori: {metadata.get('kategori', 'N/A')}")
            print(f"Source: {metadata.get('source', 'N/A')}")
            print(f"Content: {document}")
            print(f"{'-'*30}")
            
    except Exception as e:
        print(f"Error saat mencari: {e}")

def show_sample_documents(results, n_samples=10):
    """Menampilkan beberapa contoh dokumen"""
    print(f"\n{'='*50}")
    print(f"CONTOH {n_samples} DOKUMEN")
    print(f"{'='*50}")
    
    import random
    samples = random.sample(list(zip(
        results['ids'], 
        results['documents'], 
        results['metadatas']
    )), min(n_samples, len(results['ids'])))
    
    for i, (doc_id, document, metadata) in enumerate(samples, 1):
        print(f"\n[{i}] ID: {doc_id}")
        print(f"Kategori: {metadata.get('kategori', 'N/A')}")
        print(f"Source: {metadata.get('source', 'N/A')}")
        print(f"Content Preview: {document[:150]}...")
        print(f"{'-'*30}")

def main():
    """Fungsi utama"""
    while True:
        print("\n" + "="*50)
        print("CHROMADB VECTOR DATABASE EXPLORER")
        print("="*50)
        print("1. Lihat informasi semua collections")
        print("2. Lihat detail knowledge_base collection")
        print("3. Cari dokumen")
        print("4. Keluar")
        
        pilihan = input("\nPilih menu (1-4): ").strip()
        
        if pilihan == "1":
            view_collection_info()
        elif pilihan == "2":
            view_collection_details()
        elif pilihan == "3":
            # Langsung ke search
            client = chromadb.PersistentClient(path=CHROMA_DIR)
            try:
                collection = client.get_collection(name=COLLECTION_NAME)
                search_documents(collection)
            except Exception as e:
                print(f"Error: {e}")
        elif pilihan == "4":
            print("Terima kasih telah menggunakan ChromaDB Explorer!")
            break
        else:
            print("Pilihan tidak valid!")

if __name__ == "__main__":
    main()