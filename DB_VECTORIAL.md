# Guía Rápida de Bases de Datos Vectoriales para el Reto Técnico

Esta guía contiene plantillas de código listas para copiar y pegar para las bases de datos vectoriales locales más comunes en entornos de desarrollo rápido (24h).

---

## 1. Conceptos Matemáticos Clave (¡Muy valorado en entrevistas!)

Cuando indexas vectores, debes elegir una métrica de distancia:
1. **L2 (Distancia Euclidiana)**: Mide la distancia física entre dos puntos. Menor distancia = mayor similitud. En FAISS: `IndexFlatL2`.
2. **IP (Inner Product / Producto Punto)**: Mide la proyección de un vector sobre otro. Mayor producto punto = mayor similitud. Útil si los vectores ya están normalizados. En FAISS: `IndexFlatIP`.
3. **Similitud de Coseno**: Mide el ángulo entre vectores, ignorando su magnitud.
   > ⚠️ **Truco técnico para FAISS**: FAISS no tiene un índice de "Coseno" directo. Para obtener similitud de coseno, debes **normalizar L2** tus vectores antes de añadirlos a un índice `IndexFlatIP`.
   > ```python
   > import numpy as np
   > # Normalización L2 en NumPy
   > vector = vector / np.linalg.norm(vector, axis=1, keepdims=True)
   > ```

---

## 2. FAISS (Local y ultra-rápido en memoria)

FAISS es la librería de Facebook para búsqueda de similitud densa. Es ideal para pruebas locales porque no requiere levantar servidores.

### Enfoque Nativo (NumPy Puro)
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import faiss
import numpy as np

# 1. Configurar dimensiones y datos de prueba
dimension = 384  # Dimensión común (ej. all-MiniLM-L6-v2)
num_docs = 1000
vectors = np.random.random((num_docs, dimension)).astype('float32')

# 2. Normalizar para usar Similitud de Coseno
vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

# 3. Crear el índice (Inner Product para vectores normalizados)
index = faiss.IndexFlatIP(dimension)
index.add(vectors)
print(f"Vectores indexados: {index.ntotal}")

# 4. Realizar una consulta
query_vector = np.random.random((1, dimension)).astype('float32')
query_vector = query_vector / np.linalg.norm(query_vector)

k = 5  # Top K resultados
scores, indices = index.search(query_vector, k)

print("Puntajes (Similitud de Coseno):", scores[0])
print("Índices de documentos más cercanos:", indices[0])

# 5. Guardar y Cargar el índice localmente
faiss.write_index(index, "mi_indice.faiss")
# Para cargar:
# index_cargado = faiss.read_index("mi_indice.faiss")
```

### Enfoque LangChain
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 1. Crear embeddings y documentos
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs = [
    Document(page_content="El aprendizaje supervisado usa datos etiquetados.", metadata={"source": "doc1"}),
    Document(page_content="El aprendizaje no supervisado busca patrones.", metadata={"source": "doc2"}),
]

# 2. Crear base de datos vectorial
db = FAISS.from_documents(docs, embeddings)

# 3. Consultar
query = "¿Qué es el aprendizaje supervisado?"
resultados = db.similarity_search(query, k=1)
print("Resultado:", resultados[0].page_content)

# 4. Guardar y Cargar en Disco
db.save_local("faiss_index_local")
# Para cargar:
# db_cargado = FAISS.load_local("faiss_index_local", embeddings, allow_dangerous_deserialization=True)
```

---

## 3. ChromaDB (Base de datos vectorial local con persistencia SQLite)

ChromaDB es muy popular para prototipos RAG rápidos por su facilidad de uso y persistencia en disco integrada.

### Instalación rápida
```bash
pip install chromadb
```

### Código Base (Nativo de Chroma)
```python
import chromadb
from chromadb.utils import embedding_functions

# 1. Inicializar cliente con persistencia en disco
client = chromadb.PersistentClient(path="./chroma_db")

# 2. Elegir función de embeddings (HuggingFace por defecto)
# Ojo: Requiere tener sentence-transformers instalado
emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# 3. Crear o cargar una colección
collection = client.get_or_create_collection(name="documentos_soporte", embedding_function=emb_fn)

# 4. Agregar documentos (calcula embeddings automáticamente)
collection.add(
    documents=["El horario de soporte es de 7:30 a 16:30.", "Oficinas ubicadas en Tenerife."],
    metadatas=[{"categoria": "soporte"}, {"categoria": "ubicacion"}],
    ids=["id1", "id2"]
)

# 5. Realizar consulta
resultados = collection.query(
    query_texts=["¿Dónde están las oficinas?"],
    n_results=1
)
print("Resultados de consulta:", resultados['documents'][0])
```
