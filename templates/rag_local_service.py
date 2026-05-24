"""
Template: Servicio RAG Local Adaptado (FAISS + Chunker de D4-Ticket-AI)
Este servicio adapta la lógica de scraping y chunking de tu proyecto final (D4-Ticket-AI)
para trabajar con una base de datos vectorial local (FAISS) en lugar de PostgreSQL/pgvector.
"""

import os
import sys
import asyncio
from typing import List, Dict, Any, Optional

# Workarounds de OpenMP para evitar segfaults en macOS con PyTorch/FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# Librerías necesarias (se asume que ya están instaladas en el .venv)
import numpy as np
import faiss
import trafilatura
from sentence_transformers import SentenceTransformer

# Configuración de tamaño de chunk y solapamiento (estándar de tu DAW)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


class LocalRAGService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Inicializa el modelo de embeddings local y el almacén en memoria."""
        print(f"[*] Cargando modelo de embeddings local: {model_name}...")
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_embedding_dimension()

        
        # Estructuras de almacenamiento local
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product para similitud de coseno
        self.chunks_metadata: List[Dict[str, Any]] = []  # Almacena texto y origen

    def _chunk_text(self, text: str) -> List[str]:
        """
        Algoritmo de troceado heredado de D4-Ticket-AI.
        Divide el texto respetando los límites de párrafos e introduce solapamientos.
        """
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunks: List[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 <= CHUNK_SIZE:
                current = f"{current}\n\n{para}".strip() if current else para
            else:
                if current:
                    chunks.append(current)
                # Si el párrafo es más grande que CHUNK_SIZE, se realiza un corte duro
                while len(para) > CHUNK_SIZE:
                    chunks.append(para[:CHUNK_SIZE])
                    para = para[CHUNK_SIZE - CHUNK_OVERLAP:]
                current = para

        if current:
            chunks.append(current)

        return chunks

    async def scrape_and_ingest(self, url: str) -> int:
        """
        Scrapea una URL de forma asíncrona usando trafilatura (de tu DAW)
        y la indexa en el FAISS local.
        """
        print(f"[*] Descargando contenido de: {url}...")
        
        # Ejecutar trafilatura en un hilo secundario para no bloquear el bucle de eventos
        raw_html = await asyncio.to_thread(trafilatura.fetch_url, url)
        if not raw_html:
            raise ValueError(f"No se pudo descargar la URL: {url}")

        text = await asyncio.to_thread(
            trafilatura.extract,
            raw_html,
            include_comments=False,
            include_tables=True,
        )
        if not text:
            raise ValueError(f"No se pudo extraer texto limpio de: {url}")

        # Trocear el texto extraído
        chunks = self._chunk_text(text)
        if not chunks:
            return 0

        # Generar embeddings de forma síncrona/asíncrona local
        print(f"[*] Calculando embeddings para {len(chunks)} chunks locales...")
        embeddings_raw = self.embedding_model.encode(chunks, show_progress_bar=False)
        
        # Normalizar L2 para simular similitud de coseno mediante producto interno
        embeddings = embeddings_raw / np.linalg.norm(embeddings_raw, axis=1, keepdims=True)

        # Agregar al índice FAISS y a los metadatos
        self.index.add(embeddings.astype('float32'))
        for idx, chunk in enumerate(chunks):
            self.chunks_metadata.append({
                "text": chunk,
                "url": url,
                "chunk_index": idx
            })
            
        print(f"[+] Indexados {len(chunks)} chunks con éxito.")
        return len(chunks)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Busca los k documentos más relevantes a la consulta usando el índice FAISS."""
        if self.index.ntotal == 0:
            print("[!] Advertencia: La base de datos vectorial está vacía.")
            return []

        # Generar y normalizar embedding de la query
        query_emb = self.embedding_model.encode([query], show_progress_bar=False)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)

        # Realizar búsqueda
        scores, indices = self.index.search(query_emb.astype('float32'), k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS devuelve -1 si no hay suficientes elementos
                continue
            metadata = self.chunks_metadata[idx]
            results.append({
                "text": metadata["text"],
                "url": metadata["url"],
                "score": float(score)
            })
        return results

    def save(self, folder_path: str):
        """Guarda el índice FAISS y los metadatos de texto a disco."""
        os.makedirs(folder_path, exist_ok=True)
        # Guardar índice binario
        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))
        # Guardar metadatos (usando numpy para persistencia rápida)
        np.save(os.path.join(folder_path, "metadata.npy"), self.chunks_metadata)
        print(f"[+] Almacén RAG guardado localmente en: {folder_path}")

    def load(self, folder_path: str):
        """Carga el índice FAISS y los metadatos desde el disco."""
        self.index = faiss.read_index(os.path.join(folder_path, "index.faiss"))
        self.chunks_metadata = np.load(os.path.join(folder_path, "metadata.npy"), allow_pickle=True).tolist()
        print(f"[+] Almacén RAG cargado. Vectores activos: {self.index.ntotal}")


# ==========================================
# PRUEBA DE FUNCIONAMIENTO
# ==========================================
async def test_main():
    rag = LocalRAGService()
    
    # Ingerir una página web pública de ejemplo (Wikipedia o documentación)
    test_url = "https://es.wikipedia.org/wiki/Aprendizaje_autom%C3%A1tico"
    try:
        chunks_creados = await rag.scrape_and_ingest(test_url)
        print(f"Indexados: {chunks_creados} chunks.")
        
        # Realizar consulta semántica
        query = "¿Qué algoritmos se usan en el aprendizaje automático?"
        resultados = rag.retrieve(query, k=2)
        
        print("\n--- RESULTADO DE LA BÚSQUEDA ---")
        for i, res in enumerate(resultados):
            print(f"\n[{i+1}] Similitud: {res['score']:.4f} (Fuente: {res['url']})")
            print(res['text'][:300] + "...")
            
        # Guardar y Cargar Prueba
        rag.save("./mi_indice_local")
        
        # Crear nueva instancia y cargar
        nuevo_rag = LocalRAGService()
        nuevo_rag.load("./mi_indice_local")
        re_search = nuevo_rag.retrieve(query, k=1)
        print(f"\n[✓] Recarga verificada. Primer resultado recargado:\n{re_search[0]['text'][:150]}...")
        
    except Exception as e:
        print(f"[!] Error durante la prueba: {e}")
        
    # Limpieza local
    for file in ["index.faiss", "metadata.npy"]:
        path = os.path.join("./mi_indice_local", file)
        if os.path.exists(path):
            os.remove(path)
    if os.path.exists("./mi_indice_local"):
        os.rmdir("./mi_indice_local")


if __name__ == "__main__":
    # Para ejecutarlo de forma aislada: python templates/rag_local_service.py
    asyncio.run(test_main())
