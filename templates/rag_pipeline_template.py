"""
Template: Pipeline de RAG (Retrieval-Augmented Generation)
Este script contiene dos enfoques para resolver un problema RAG:
1. Enfoque Modular con LangChain y FAISS (Estándar de la industria).
2. Enfoque Nativo con NumPy/SciPy (Demuestra entendimiento matemático de embeddings y similitud).
"""

import os
import sys

# Workarounds de OpenMP para evitar segfaults en macOS con PyTorch/FAISS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from typing import List, Dict, Any
import numpy as np


# ==========================================
# ENFOQUE 1: LANGCHAIN + FAISS
# ==========================================

try:
    from langchain_community.document_loaders import TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    # Para usar OpenAI si dispones de API Key:
    # from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class LangChainRAGPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if not HAS_LANGCHAIN:
            raise ImportError("Instala langchain, langchain-community y faiss-cpu para usar este pipeline.")
        
        print(f"[*] Inicializando embeddings con: {model_name}")
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        self.vector_store = None

    def ingest_documents(self, file_path: str):
        """Lee un archivo de texto, lo divide en chunks y crea la DB vectorial."""
        print(f"[*] Cargando documento desde: {file_path}")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        chunks = self.text_splitter.split_documents(documents)
        print(f"[+] Documento dividido en {len(chunks)} chunks.")
        
        print("[*] Generando embeddings e indexando en FAISS...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("[+] Indexación completada con éxito.")

    def query(self, question: str, llm_runner) -> str:
        """Realiza la recuperación y genera respuesta usando un LLM provisto (ej. Ollama o OpenAI)."""
        if not self.vector_store:
            raise ValueError("Primero debes ingerir documentos con ingest_documents().")
            
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Definición del Prompt estándar
        template = """Usa el siguiente contexto para responder a la pregunta del final. 
Si no sabes la respuesta, di que no la sabes, no intentes inventar una respuesta.

Contexto:
{context}

Pregunta: {question}

Respuesta útil y detallada:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Helper para formatear los documentos recuperados
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # Pipeline declarativo LCEL (LangChain Expression Language)
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm_runner
            | StrOutputParser()
        )
        
        return rag_chain.invoke(question)


# ==========================================
# ENFOQUE 2: RAG NATIVO (NumPy + SentenceTransformers)
# ==========================================

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class NativeRAGPipeline:
    """
    RAG ligero implementado desde cero. Demuestra a los evaluadores que
    comprendes cómo funciona la similitud de coseno y la indexación sin abstracciones.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if not HAS_SENTENCE_TRANSFORMERS:
            print("[!] Advertencia: sentence-transformers no está instalado.")
            self.model = None
        else:
            self.model = SentenceTransformer(model_name)
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.array([])

    def split_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Divide el texto de manera simple usando solapamiento de caracteres."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def ingest_text(self, text: str):
        if not self.model:
            raise ImportError("Instala sentence-transformers para calcular embeddings.")
        
        self.chunks = self.split_text(text)
        print(f"[*] Generando embeddings para {len(self.chunks)} chunks nativos...")
        # Generar embeddings y normalizarlos para similitud de coseno directa (producto punto)
        embeddings_raw = self.model.encode(self.chunks, show_progress_bar=False)
        self.embeddings = embeddings_raw / np.linalg.norm(embeddings_raw, axis=1, keepdims=True)
        print("[+] Indexación nativa completada.")

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Recupera los K chunks más relevantes usando producto punto sobre embeddings normalizados."""
        if self.embeddings.size == 0:
            raise ValueError("No hay datos ingeridos.")
            
        # Generar y normalizar embedding de la consulta
        query_emb = self.model.encode([query], show_progress_bar=False)[0]
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        # Calcular similitudes (producto punto)
        similarities = np.dot(self.embeddings, query_emb)
        
        # Obtener los índices de los top K de mayor a menor
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return results


# ==========================================
# EJEMPLO DE USO (PRUEBA LOCAL)
# ==========================================

if __name__ == "__main__":
    # Crear un archivo de prueba temporal
    temp_file = "temp_data.txt"
    sample_text = (
        "El aprendizaje supervisado es un tipo de aprendizaje automático donde el modelo "
        "se entrena con datos etiquetados. Los algoritmos comunes incluyen regresión lineal, "
        "bosques aleatorios y redes neuronales.\n\n"
        "El aprendizaje no supervisado trabaja con datos no etiquetados, buscando patrones ocultos. "
        "Ejemplos comunes son K-means y algoritmos de clustering jerárquico.\n\n"
        "RAG (Generación Aumentada por Recuperación) combina la recuperación de información con modelos "
        "generativos de lenguaje para proporcionar respuestas más precisas y actualizadas basadas en "
        "documentos de referencia externos."
    )
    
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(sample_text)
        
    print("=== PROBANDO RAG NATIVO (NumPy) ===")
    if HAS_SENTENCE_TRANSFORMERS:
        rag = NativeRAGPipeline()
        rag.ingest_text(sample_text)
        query_str = "¿Qué es RAG y cómo funciona?"
        retrieved_docs = rag.retrieve(query_str, k=1)
        
        print(f"\nPregunta: {query_str}")
        print(f"Documento Recuperado (Score: {retrieved_docs[0]['score']:.4f}):")
        print(retrieved_docs[0]['text'])
    else:
        print("[!] Sáltate la prueba nativa: instala `pip install sentence-transformers` primero.")

    # Limpiar archivo temporal
    if os.path.exists(temp_file):
        os.remove(temp_file)
